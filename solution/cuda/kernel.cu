/*
 * CUDA Fused MoE kernel for FlashInfer MLSys 2026 Contest Track A.
 * cuBLAS with TF32 tensor cores + multi-stream expert parallelism.
 * Target: NVIDIA B200 (sm_100). Output: bfloat16.
 */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublas_v2.h>
#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <cfloat>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <dlpack/dlpack.h>
using tvm::ffi::TensorView;
static constexpr int H=7168,I=2048,G1=4096,NE=256,NL=32,TK=8,NG=8,TG=4,QB=128;
static constexpr int GS=NE/NG,NHB=H/QB,NIB=I/QB,NG1B=G1/QB,MC=512,NSTREAMS=4;
static inline cudaStream_t get_stream(){int d;cudaGetDevice(&d);return(cudaStream_t)TVMFFIEnvGetStream(kDLCUDA,d);}
__device__ __forceinline__ float fp8f(uint8_t v){__nv_fp8_e4m3 f;f.__x=v;return float(f);}
using PFN_C=cublasStatus_t(*)(cublasHandle_t*);using PFN_SS=cublasStatus_t(*)(cublasHandle_t,cudaStream_t);
using PFN_SM=cublasStatus_t(*)(cublasHandle_t,cublasMath_t);
using PFN_SG=cublasStatus_t(*)(cublasHandle_t,cublasOperation_t,cublasOperation_t,int,int,int,const float*,const float*,int,const float*,int,const float*,float*,int);
static cublasHandle_t s_h[NSTREAMS]={};static PFN_SG s_sg=nullptr;static PFN_SS s_ss=nullptr;
static cudaStream_t s_st[NSTREAMS]={};static cudaEvent_t s_re=nullptr,s_se[NSTREAMS]={};static bool s_i=false;
static void cb_init(){if(s_i)return;void*lib=dlopen("libcublas.so.12",RTLD_NOW|RTLD_GLOBAL);if(!lib)lib=dlopen("libcublas.so",RTLD_NOW|RTLD_GLOBAL);if(!lib)return;auto C=reinterpret_cast<PFN_C>(dlsym(lib,"cublasCreate_v2"));s_ss=reinterpret_cast<PFN_SS>(dlsym(lib,"cublasSetStream_v2"));s_sg=reinterpret_cast<PFN_SG>(dlsym(lib,"cublasSgemm_v2"));auto sm=reinterpret_cast<PFN_SM>(dlsym(lib,"cublasSetMathMode"));for(int i=0;i<NSTREAMS;i++){if(C)C(&s_h[i]);if(sm&&s_h[i])sm(s_h[i],CUBLAS_TF32_TENSOR_OP_MATH);cudaStreamCreate(&s_st[i]);cudaEventCreateWithFlags(&s_se[i],cudaEventDisableTiming);}cudaEventCreateWithFlags(&s_re,cudaEventDisableTiming);s_i=true;}
__global__ void routing_kernel(const float*__restrict__ lg,const __nv_bfloat16*__restrict__ bi,int*__restrict__ ti,float*__restrict__ tw,int S,float sf){int t=threadIdx.x,tok=blockIdx.x;if(tok>=S)return;float s=1.0f/(1.0f+expf(-lg[tok*NE+t]));float sb=s+__bfloat162float(bi[t]);__shared__ float ss[NE],sg[NE];ss[t]=sb;sg[t]=s;__syncthreads();if(t==0){float gs[NG];for(int g=0;g<NG;g++){int b=g*GS;float m1=-FLT_MAX,m2=-FLT_MAX;for(int i=0;i<GS;i++){float v=ss[b+i];if(v>m1){m2=m1;m1=v;}else if(v>m2)m2=v;}gs[g]=m1+m2;}bool gk[NG]={};for(int k=0;k<TG;k++){float bv=-FLT_MAX;int bg=0;for(int g=0;g<NG;g++)if(!gk[g]&&gs[g]>bv){bv=gs[g];bg=g;}gk[bg]=true;}float sp[NE];for(int e=0;e<NE;e++)sp[e]=gk[e/GS]?ss[e]:-FLT_MAX;int sel[TK];for(int k=0;k<TK;k++){float bv=-FLT_MAX;int bi2=0;for(int e=0;e<NE;e++)if(sp[e]>bv){bv=sp[e];bi2=e;}sel[k]=bi2;sp[bi2]=-FLT_MAX;}float ws=1e-20f;for(int k=0;k<TK;k++)ws+=sg[sel[k]];for(int k=0;k<TK;k++){ti[tok*TK+k]=sel[k];tw[tok*TK+k]=sg[sel[k]]/ws*sf;}}}
__global__ void count_k(const int*ti,int*ec,int S,int leo){int tok=blockIdx.x;if(tok>=S)return;int k=threadIdx.x;if(k>=TK)return;int le=ti[tok*TK+k]-leo;if(le>=0&&le<NL)atomicAdd(&ec[le],1);}
__global__ void prefix_sum_k(const int*c,int*o){if(threadIdx.x!=0)return;int s=0;for(int i=0;i<NL;i++){o[i]=s;s+=c[i];}o[NL]=s;}
__global__ void scatter_k(const int*ti,const float*tw,int*toi,float*tow,const int*eo,int*ep,int S,int leo){int tok=blockIdx.x;if(tok>=S)return;int k=threadIdx.x;if(k>=TK)return;int le=ti[tok*TK+k]-leo;if(le>=0&&le<NL){int p=atomicAdd(&ep[le],1);toi[eo[le]+p]=tok;tow[eo[le]+p]=tw[tok*TK+k];}}
__global__ void dequant_w(const uint8_t*w,const float*sc,float*out,int N,int K,int nkb){int idx=blockIdx.x*blockDim.x+threadIdx.x;if(idx>=N*K)return;out[idx]=fp8f(w[idx])*sc[(idx/K/QB)*nkb+(idx%K)/QB];}
__global__ void fused_dequant_gather(const uint8_t*hs,const float*sc,const int*ti,float*out,int M,int as0,int as1){int idx=blockIdx.x*blockDim.x+threadIdx.x;if(idx>=M*H)return;int m=idx/H,h=idx%H;int tok=ti[m];out[idx]=fp8f(hs[(int64_t)tok*H+h])*sc[(h/QB)*as0+tok*as1];}
__global__ void swiglu_k(const float*g1,float*out,int M){int idx=blockIdx.x*blockDim.x+threadIdx.x;if(idx>=M*I)return;int m=idx/I,i=idx%I;float x1=g1[m*G1+i],x2=g1[m*G1+I+i];out[m*I+i]=x2/(1.0f+expf(-x2))*x1;}
__global__ void accum_k(const float*g2,const int*ti,const float*tw,float*out,int M){int m=blockIdx.y;if(m>=M)return;int h=blockIdx.x*blockDim.x+threadIdx.x;if(h>=H)return;atomicAdd(&out[ti[m]*H+h],g2[m*H+h]*tw[m]);}
__global__ void f2b(const float*in,__nv_bfloat16*out,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=__float2bfloat16(in[i]);}
struct WS{void*p=nullptr;size_t sz=0;void ensure(size_t n){if(n<=sz&&p)return;if(p)cudaFree(p);cudaMalloc(&p,n);sz=n;}};
static WS g_ws;
void KernelFunc(TensorView routing_logits,TensorView routing_bias,TensorView hidden_states,TensorView hidden_states_scale,TensorView gemm1_weights,TensorView gemm1_weights_scale,TensorView gemm2_weights,TensorView gemm2_weights_scale,int64_t local_expert_offset,double routed_scaling_factor,TensorView output){
    cudaStream_t stream=get_stream();cb_init();int S=(int)routing_logits.size(0);
    auto*logp=static_cast<const float*>(routing_logits.data_ptr());auto*biap=static_cast<const __nv_bfloat16*>(routing_bias.data_ptr());auto*hsp=static_cast<const uint8_t*>(hidden_states.data_ptr());auto*hssp=static_cast<const float*>(hidden_states_scale.data_ptr());auto*g1wp=static_cast<const uint8_t*>(gemm1_weights.data_ptr());auto*g1sp=static_cast<const float*>(gemm1_weights_scale.data_ptr());auto*g2wp=static_cast<const uint8_t*>(gemm2_weights.data_ptr());auto*g2sp=static_cast<const float*>(gemm2_weights_scale.data_ptr());auto*outp=static_cast<__nv_bfloat16*>(output.data_ptr());
    int as0=(int)hidden_states_scale.stride(0),as1=(int)hidden_states_scale.stride(1),leo=(int)local_expert_offset;int64_t tot=(int64_t)S*TK;
    size_t n=0;size_t o_tki=n;n+=tot*4;size_t o_tkw=n;n+=tot*4;size_t o_ec=n;n+=NL*4;size_t o_eod=n;n+=(NL+1)*4;size_t o_ep=n;n+=NL*4;size_t o_toi=n;n+=tot*4;size_t o_tow=n;n+=tot*4;size_t o_ofp=n;n+=(int64_t)S*H*4;
    size_t ps=0;size_t pw1=ps;ps+=(int64_t)G1*H*4;size_t pw2=ps;ps+=(int64_t)H*I*4;size_t pag=ps;ps+=(int64_t)MC*H*4;size_t pg1=ps;ps+=(int64_t)MC*G1*4;size_t psg=ps;ps+=(int64_t)MC*I*4;size_t pg2=ps;ps+=(int64_t)MC*H*4;
    size_t o_st=n;n+=ps*NSTREAMS;g_ws.ensure(n);char*b=(char*)g_ws.p;
    int*tki=(int*)(b+o_tki);float*tkw=(float*)(b+o_tkw);int*ec=(int*)(b+o_ec);int*eod=(int*)(b+o_eod);int*ep=(int*)(b+o_ep);int*toi=(int*)(b+o_toi);float*tow=(float*)(b+o_tow);float*ofp=(float*)(b+o_ofp);
    cudaMemsetAsync(ofp,0,(int64_t)S*H*4,stream);
    routing_kernel<<<S,NE,0,stream>>>(logp,biap,tki,tkw,S,(float)routed_scaling_factor);
    cudaMemsetAsync(ec,0,NL*4,stream);count_k<<<S,TK,0,stream>>>(tki,ec,S,leo);
    prefix_sum_k<<<1,1,0,stream>>>(ec,eod);cudaMemsetAsync(ep,0,NL*4,stream);scatter_k<<<S,TK,0,stream>>>(tki,tkw,toi,tow,eod,ep,S,leo);
    int ho[NL+1];cudaMemcpyAsync(ho,eod,(NL+1)*4,cudaMemcpyDeviceToHost,stream);cudaStreamSynchronize(stream);
    if(ho[NL]==0){int nn=S*H;f2b<<<(nn+255)/256,256,0,stream>>>(ofp,outp,nn);cudaStreamSynchronize(stream);return;}
    cudaEventRecord(s_re,stream);bool sw[NSTREAMS]={};
    for(int e=0;e<NL;e++){int st=ho[e],Me=ho[e+1]-st;if(!Me)continue;int sid=e%NSTREAMS;cudaStream_t ws=s_st[sid];char*sb=b+o_st+ps*sid;
        float*w1f=(float*)(sb+pw1);float*w2f=(float*)(sb+pw2);float*ag=(float*)(sb+pag);float*g1b=(float*)(sb+pg1);float*sgb=(float*)(sb+psg);float*g2b=(float*)(sb+pg2);
        if(!sw[sid]){cudaStreamWaitEvent(ws,s_re);s_ss(s_h[sid],ws);sw[sid]=true;}
        {int nn=G1*H;dequant_w<<<(nn+255)/256,256,0,ws>>>(g1wp+(int64_t)e*G1*H,g1sp+(int64_t)e*NG1B*NHB,w1f,G1,H,NHB);}
        {int nn=H*I;dequant_w<<<(nn+255)/256,256,0,ws>>>(g2wp+(int64_t)e*H*I,g2sp+(int64_t)e*NHB*NIB,w2f,H,I,NIB);}
        for(int c0=0;c0<Me;c0+=MC){int M=(Me-c0<MC)?Me-c0:MC;int*eti=toi+st+c0;float*etw=tow+st+c0;
            {int nn=M*H;fused_dequant_gather<<<(nn+255)/256,256,0,ws>>>(hsp,hssp,eti,ag,M,as0,as1);}
            {float a=1,b2=0;s_sg(s_h[sid],CUBLAS_OP_T,CUBLAS_OP_N,G1,M,H,&a,w1f,H,ag,H,&b2,g1b,G1);}
            {int nn=M*I;swiglu_k<<<(nn+255)/256,256,0,ws>>>(g1b,sgb,M);}
            {float a=1,b2=0;s_sg(s_h[sid],CUBLAS_OP_T,CUBLAS_OP_N,H,M,I,&a,w2f,I,sgb,I,&b2,g2b,H);}
            {dim3 grid((H+255)/256,M);accum_k<<<grid,256,0,ws>>>(g2b,eti,etw,ofp,M);}}}
    for(int i=0;i<NSTREAMS;i++){cudaEventRecord(s_se[i],s_st[i]);cudaStreamWaitEvent(stream,s_se[i]);}
    {int nn=S*H;f2b<<<(nn+255)/256,256,0,stream>>>(ofp,outp,nn);}cudaStreamSynchronize(stream);}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel,KernelFunc);
