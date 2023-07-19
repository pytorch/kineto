/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "cupti_strings.h"

namespace libkineto {

const char* memcpyKindString(
    CUpti_ActivityMemcpyKind kind) {
  switch (kind) {
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
      return "HtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
      return "DtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
      return "HtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
      return "AtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
      return "AtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
      return "AtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
      return "DtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
      return "DtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
      return "HtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP:
      return "PtoP";
    default:
      break;
  }
  return "<unknown>";
}

const char* memoryKindString(
    CUpti_ActivityMemoryKind kind) {
  switch (kind) {
    case CUPTI_ACTIVITY_MEMORY_KIND_UNKNOWN:
      return "Unknown";
    case CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE:
      return "Pageable";
    case CUPTI_ACTIVITY_MEMORY_KIND_PINNED:
      return "Pinned";
    case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE:
      return "Device";
    case CUPTI_ACTIVITY_MEMORY_KIND_ARRAY:
      return "Array";
    case CUPTI_ACTIVITY_MEMORY_KIND_MANAGED:
      return "Managed";
    case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC:
      return "Device Static";
    case CUPTI_ACTIVITY_MEMORY_KIND_MANAGED_STATIC:
      return "Managed Static";
    case CUPTI_ACTIVITY_MEMORY_KIND_FORCE_INT:
      return "Force Int";
    default:
      return "Unrecognized";
  }
}

const char* overheadKindString(
    CUpti_ActivityOverheadKind kind) {
  switch (kind) {
    case CUPTI_ACTIVITY_OVERHEAD_UNKNOWN:
      return "Unknown";
    case CUPTI_ACTIVITY_OVERHEAD_DRIVER_COMPILER:
      return "Driver Compiler";
    case CUPTI_ACTIVITY_OVERHEAD_CUPTI_BUFFER_FLUSH:
      return "Buffer Flush";
    case CUPTI_ACTIVITY_OVERHEAD_CUPTI_INSTRUMENTATION:
      return "Instrumentation";
    case CUPTI_ACTIVITY_OVERHEAD_CUPTI_RESOURCE:
      return "Resource";
    case CUPTI_ACTIVITY_OVERHEAD_FORCE_INT:
      return "Force Int";
    default:
      return "Unrecognized";
  }
}



static const char* runtimeCbidNames[] = {
    "INVALID",
    "cudaDriverGetVersion",
    "cudaRuntimeGetVersion",
    "cudaGetDeviceCount",
    "cudaGetDeviceProperties",
    "cudaChooseDevice",
    "cudaGetChannelDesc",
    "cudaCreateChannelDesc",
    "cudaConfigureCall",
    "cudaSetupArgument",
    "cudaGetLastError",
    "cudaPeekAtLastError",
    "cudaGetErrorString",
    "cudaLaunch",
    "cudaFuncSetCacheConfig",
    "cudaFuncGetAttributes",
    "cudaSetDevice",
    "cudaGetDevice",
    "cudaSetValidDevices",
    "cudaSetDeviceFlags",
    "cudaMalloc",
    "cudaMallocPitch",
    "cudaFree",
    "cudaMallocArray",
    "cudaFreeArray",
    "cudaMallocHost",
    "cudaFreeHost",
    "cudaHostAlloc",
    "cudaHostGetDevicePointer",
    "cudaHostGetFlags",
    "cudaMemGetInfo",
    "cudaMemcpy",
    "cudaMemcpy2D",
    "cudaMemcpyToArray",
    "cudaMemcpy2DToArray",
    "cudaMemcpyFromArray",
    "cudaMemcpy2DFromArray",
    "cudaMemcpyArrayToArray",
    "cudaMemcpy2DArrayToArray",
    "cudaMemcpyToSymbol",
    "cudaMemcpyFromSymbol",
    "cudaMemcpyAsync",
    "cudaMemcpyToArrayAsync",
    "cudaMemcpyFromArrayAsync",
    "cudaMemcpy2DAsync",
    "cudaMemcpy2DToArrayAsync",
    "cudaMemcpy2DFromArrayAsync",
    "cudaMemcpyToSymbolAsync",
    "cudaMemcpyFromSymbolAsync",
    "cudaMemset",
    "cudaMemset2D",
    "cudaMemsetAsync",
    "cudaMemset2DAsync",
    "cudaGetSymbolAddress",
    "cudaGetSymbolSize",
    "cudaBindTexture",
    "cudaBindTexture2D",
    "cudaBindTextureToArray",
    "cudaUnbindTexture",
    "cudaGetTextureAlignmentOffset",
    "cudaGetTextureReference",
    "cudaBindSurfaceToArray",
    "cudaGetSurfaceReference",
    "cudaGLSetGLDevice",
    "cudaGLRegisterBufferObject",
    "cudaGLMapBufferObject",
    "cudaGLUnmapBufferObject",
    "cudaGLUnregisterBufferObject",
    "cudaGLSetBufferObjectMapFlags",
    "cudaGLMapBufferObjectAsync",
    "cudaGLUnmapBufferObjectAsync",
    "cudaWGLGetDevice",
    "cudaGraphicsGLRegisterImage",
    "cudaGraphicsGLRegisterBuffer",
    "cudaGraphicsUnregisterResource",
    "cudaGraphicsResourceSetMapFlags",
    "cudaGraphicsMapResources",
    "cudaGraphicsUnmapResources",
    "cudaGraphicsResourceGetMappedPointer",
    "cudaGraphicsSubResourceGetMappedArray",
    "cudaVDPAUGetDevice",
    "cudaVDPAUSetVDPAUDevice",
    "cudaGraphicsVDPAURegisterVideoSurface",
    "cudaGraphicsVDPAURegisterOutputSurface",
    "cudaD3D11GetDevice",
    "cudaD3D11GetDevices",
    "cudaD3D11SetDirect3DDevice",
    "cudaGraphicsD3D11RegisterResource",
    "cudaD3D10GetDevice",
    "cudaD3D10GetDevices",
    "cudaD3D10SetDirect3DDevice",
    "cudaGraphicsD3D10RegisterResource",
    "cudaD3D10RegisterResource",
    "cudaD3D10UnregisterResource",
    "cudaD3D10MapResources",
    "cudaD3D10UnmapResources",
    "cudaD3D10ResourceSetMapFlags",
    "cudaD3D10ResourceGetSurfaceDimensions",
    "cudaD3D10ResourceGetMappedArray",
    "cudaD3D10ResourceGetMappedPointer",
    "cudaD3D10ResourceGetMappedSize",
    "cudaD3D10ResourceGetMappedPitch",
    "cudaD3D9GetDevice",
    "cudaD3D9GetDevices",
    "cudaD3D9SetDirect3DDevice",
    "cudaD3D9GetDirect3DDevice",
    "cudaGraphicsD3D9RegisterResource",
    "cudaD3D9RegisterResource",
    "cudaD3D9UnregisterResource",
    "cudaD3D9MapResources",
    "cudaD3D9UnmapResources",
    "cudaD3D9ResourceSetMapFlags",
    "cudaD3D9ResourceGetSurfaceDimensions",
    "cudaD3D9ResourceGetMappedArray",
    "cudaD3D9ResourceGetMappedPointer",
    "cudaD3D9ResourceGetMappedSize",
    "cudaD3D9ResourceGetMappedPitch",
    "cudaD3D9Begin",
    "cudaD3D9End",
    "cudaD3D9RegisterVertexBuffer",
    "cudaD3D9UnregisterVertexBuffer",
    "cudaD3D9MapVertexBuffer",
    "cudaD3D9UnmapVertexBuffer",
    "cudaThreadExit",
    "cudaSetDoubleForDevice",
    "cudaSetDoubleForHost",
    "cudaThreadSynchronize",
    "cudaThreadGetLimit",
    "cudaThreadSetLimit",
    "cudaStreamCreate",
    "cudaStreamDestroy",
    "cudaStreamSynchronize",
    "cudaStreamQuery",
    "cudaEventCreate",
    "cudaEventCreateWithFlags",
    "cudaEventRecord",
    "cudaEventDestroy",
    "cudaEventSynchronize",
    "cudaEventQuery",
    "cudaEventElapsedTime",
    "cudaMalloc3D",
    "cudaMalloc3DArray",
    "cudaMemset3D",
    "cudaMemset3DAsync",
    "cudaMemcpy3D",
    "cudaMemcpy3DAsync",
    "cudaThreadSetCacheConfig",
    "cudaStreamWaitEvent",
    "cudaD3D11GetDirect3DDevice",
    "cudaD3D10GetDirect3DDevice",
    "cudaThreadGetCacheConfig",
    "cudaPointerGetAttributes",
    "cudaHostRegister",
    "cudaHostUnregister",
    "cudaDeviceCanAccessPeer",
    "cudaDeviceEnablePeerAccess",
    "cudaDeviceDisablePeerAccess",
    "cudaPeerRegister",
    "cudaPeerUnregister",
    "cudaPeerGetDevicePointer",
    "cudaMemcpyPeer",
    "cudaMemcpyPeerAsync",
    "cudaMemcpy3DPeer",
    "cudaMemcpy3DPeerAsync",
    "cudaDeviceReset",
    "cudaDeviceSynchronize",
    "cudaDeviceGetLimit",
    "cudaDeviceSetLimit",
    "cudaDeviceGetCacheConfig",
    "cudaDeviceSetCacheConfig",
    "cudaProfilerInitialize",
    "cudaProfilerStart",
    "cudaProfilerStop",
    "cudaDeviceGetByPCIBusId",
    "cudaDeviceGetPCIBusId",
    "cudaGLGetDevices",
    "cudaIpcGetEventHandle",
    "cudaIpcOpenEventHandle",
    "cudaIpcGetMemHandle",
    "cudaIpcOpenMemHandle",
    "cudaIpcCloseMemHandle",
    "cudaArrayGetInfo",
    "cudaFuncSetSharedMemConfig",
    "cudaDeviceGetSharedMemConfig",
    "cudaDeviceSetSharedMemConfig",
    "cudaCreateTextureObject",
    "cudaDestroyTextureObject",
    "cudaGetTextureObjectResourceDesc",
    "cudaGetTextureObjectTextureDesc",
    "cudaCreateSurfaceObject",
    "cudaDestroySurfaceObject",
    "cudaGetSurfaceObjectResourceDesc",
    "cudaMallocMipmappedArray",
    "cudaGetMipmappedArrayLevel",
    "cudaFreeMipmappedArray",
    "cudaBindTextureToMipmappedArray",
    "cudaGraphicsResourceGetMappedMipmappedArray",
    "cudaStreamAddCallback",
    "cudaStreamCreateWithFlags",
    "cudaGetTextureObjectResourceViewDesc",
    "cudaDeviceGetAttribute",
    "cudaStreamDestroy",
    "cudaStreamCreateWithPriority",
    "cudaStreamGetPriority",
    "cudaStreamGetFlags",
    "cudaDeviceGetStreamPriorityRange",
    "cudaMallocManaged",
    "cudaOccupancyMaxActiveBlocksPerMultiprocessor",
    "cudaStreamAttachMemAsync",
    "cudaGetErrorName",
    "cudaOccupancyMaxActiveBlocksPerMultiprocessor",
    "cudaLaunchKernel",
    "cudaGetDeviceFlags",
    "cudaLaunch_ptsz",
    "cudaLaunchKernel_ptsz",
    "cudaMemcpy_ptds",
    "cudaMemcpy2D_ptds",
    "cudaMemcpyToArray_ptds",
    "cudaMemcpy2DToArray_ptds",
    "cudaMemcpyFromArray_ptds",
    "cudaMemcpy2DFromArray_ptds",
    "cudaMemcpyArrayToArray_ptds",
    "cudaMemcpy2DArrayToArray_ptds",
    "cudaMemcpyToSymbol_ptds",
    "cudaMemcpyFromSymbol_ptds",
    "cudaMemcpyAsync_ptsz",
    "cudaMemcpyToArrayAsync_ptsz",
    "cudaMemcpyFromArrayAsync_ptsz",
    "cudaMemcpy2DAsync_ptsz",
    "cudaMemcpy2DToArrayAsync_ptsz",
    "cudaMemcpy2DFromArrayAsync_ptsz",
    "cudaMemcpyToSymbolAsync_ptsz",
    "cudaMemcpyFromSymbolAsync_ptsz",
    "cudaMemset_ptds",
    "cudaMemset2D_ptds",
    "cudaMemsetAsync_ptsz",
    "cudaMemset2DAsync_ptsz",
    "cudaStreamGetPriority_ptsz",
    "cudaStreamGetFlags_ptsz",
    "cudaStreamSynchronize_ptsz",
    "cudaStreamQuery_ptsz",
    "cudaStreamAttachMemAsync_ptsz",
    "cudaEventRecord_ptsz",
    "cudaMemset3D_ptds",
    "cudaMemset3DAsync_ptsz",
    "cudaMemcpy3D_ptds",
    "cudaMemcpy3DAsync_ptsz",
    "cudaStreamWaitEvent_ptsz",
    "cudaStreamAddCallback_ptsz",
    "cudaMemcpy3DPeer_ptds",
    "cudaMemcpy3DPeerAsync_ptsz",
    "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
    "cudaMemPrefetchAsync",
    "cudaMemPrefetchAsync_ptsz",
    "cudaMemAdvise",
    "cudaDeviceGetP2PAttribute",
    "cudaGraphicsEGLRegisterImage",
    "cudaEGLStreamConsumerConnect",
    "cudaEGLStreamConsumerDisconnect",
    "cudaEGLStreamConsumerAcquireFrame",
    "cudaEGLStreamConsumerReleaseFrame",
    "cudaEGLStreamProducerConnect",
    "cudaEGLStreamProducerDisconnect",
    "cudaEGLStreamProducerPresentFrame",
    "cudaEGLStreamProducerReturnFrame",
    "cudaGraphicsResourceGetMappedEglFrame",
    "cudaMemRangeGetAttribute",
    "cudaMemRangeGetAttributes",
    "cudaEGLStreamConsumerConnectWithFlags",
    "cudaLaunchCooperativeKernel",
    "cudaLaunchCooperativeKernel_ptsz",
    "cudaEventCreateFromEGLSync",
    "cudaLaunchCooperativeKernelMultiDevice",
    "cudaFuncSetAttribute",
    "cudaImportExternalMemory",
    "cudaExternalMemoryGetMappedBuffer",
    "cudaExternalMemoryGetMappedMipmappedArray",
    "cudaDestroyExternalMemory",
    "cudaImportExternalSemaphore",
    "cudaSignalExternalSemaphoresAsync",
    "cudaSignalExternalSemaphoresAsync_ptsz",
    "cudaWaitExternalSemaphoresAsync",
    "cudaWaitExternalSemaphoresAsync_ptsz",
    "cudaDestroyExternalSemaphore",
    "cudaLaunchHostFunc",
    "cudaLaunchHostFunc_ptsz",
    "cudaGraphCreate",
    "cudaGraphKernelNodeGetParams",
    "cudaGraphKernelNodeSetParams",
    "cudaGraphAddKernelNode",
    "cudaGraphAddMemcpyNode",
    "cudaGraphMemcpyNodeGetParams",
    "cudaGraphMemcpyNodeSetParams",
    "cudaGraphAddMemsetNode",
    "cudaGraphMemsetNodeGetParams",
    "cudaGraphMemsetNodeSetParams",
    "cudaGraphAddHostNode",
    "cudaGraphHostNodeGetParams",
    "cudaGraphAddChildGraphNode",
    "cudaGraphChildGraphNodeGetGraph",
    "cudaGraphAddEmptyNode",
    "cudaGraphClone",
    "cudaGraphNodeFindInClone",
    "cudaGraphNodeGetType",
    "cudaGraphGetRootNodes",
    "cudaGraphNodeGetDependencies",
    "cudaGraphNodeGetDependentNodes",
    "cudaGraphAddDependencies",
    "cudaGraphRemoveDependencies",
    "cudaGraphDestroyNode",
    "cudaGraphInstantiate",
    "cudaGraphLaunch",
    "cudaGraphLaunch_ptsz",
    "cudaGraphExecDestroy",
    "cudaGraphDestroy",
    "cudaStreamBeginCapture",
    "cudaStreamBeginCapture_ptsz",
    "cudaStreamIsCapturing",
    "cudaStreamIsCapturing_ptsz",
    "cudaStreamEndCapture",
    "cudaStreamEndCapture_ptsz",
    "cudaGraphHostNodeSetParams",
    "cudaGraphGetNodes",
    "cudaGraphGetEdges",
    "cudaStreamGetCaptureInfo",
    "cudaStreamGetCaptureInfo_ptsz",
    "cudaGraphExecKernelNodeSetParams",
    "cudaThreadExchangeStreamCaptureMode",
    "cudaDeviceGetNvSciSyncAttributes",
    "cudaOccupancyAvailableDynamicSMemPerBlock",
    "cudaStreamSetFlags",
    "cudaStreamSetFlags_ptsz",
    "cudaGraphExecMemcpyNodeSetParams",
    "cudaGraphExecMemsetNodeSetParams",
    "cudaGraphExecHostNodeSetParams",
    "cudaGraphExecUpdate",
    "cudaGetFuncBySymbol",
    "cudaCtxResetPersistingL2Cache",
    "cudaGraphKernelNodeCopyAttributes",
    "cudaGraphKernelNodeGetAttribute",
    "cudaGraphKernelNodeSetAttribute",
    "cudaStreamCopyAttributes",
    "cudaStreamCopyAttributes_ptsz",
    "cudaStreamGetAttribute",
    "cudaStreamGetAttribute_ptsz",
    "cudaStreamSetAttribute",
    "cudaStreamSetAttribute_ptsz",
    "cudaDeviceGetTexture1DLinearMaxWidth",
    "cudaGraphUpload",
    "cudaGraphUpload_ptsz",
    "cudaGraphAddMemcpyNodeToSymbol",
    "cudaGraphAddMemcpyNodeFromSymbol",
    "cudaGraphAddMemcpyNode1D",
    "cudaGraphMemcpyNodeSetParamsToSymbol",
    "cudaGraphMemcpyNodeSetParamsFromSymbol",
    "cudaGraphMemcpyNodeSetParams1D",
    "cudaGraphExecMemcpyNodeSetParamsToSymbol",
    "cudaGraphExecMemcpyNodeSetParamsFromSymbol",
    "cudaGraphExecMemcpyNodeSetParams1D",
    "cudaArrayGetSparseProperties",
    "cudaMipmappedArrayGetSparseProperties",
    "cudaGraphExecChildGraphNodeSetParams",
    "cudaGraphAddEventRecordNode",
    "cudaGraphEventRecordNodeGetEvent",
    "cudaGraphEventRecordNodeSetEvent",
    "cudaGraphAddEventWaitNode",
    "cudaGraphEventWaitNodeGetEvent",
    "cudaGraphEventWaitNodeSetEvent",
    "cudaGraphExecEventRecordNodeSetEvent",
    "cudaGraphExecEventWaitNodeSetEvent",
    "cudaEventRecordWithFlags",
    "cudaEventRecordWithFlags_ptsz",
    "cudaDeviceGetDefaultMemPool",
    "cudaMallocAsync",
    "cudaMallocAsync_ptsz",
    "cudaFreeAsync",
    "cudaFreeAsync_ptsz",
    "cudaMemPoolTrimTo",
    "cudaMemPoolSetAttribute",
    "cudaMemPoolGetAttribute",
    "cudaMemPoolSetAccess",
    "cudaArrayGetPlane",
    "cudaMemPoolGetAccess",
    "cudaMemPoolCreate",
    "cudaMemPoolDestroy",
    "cudaDeviceSetMemPool",
    "cudaDeviceGetMemPool",
    "cudaMemPoolExportToShareableHandle",
    "cudaMemPoolImportFromShareableHandle",
    "cudaMemPoolExportPointer",
    "cudaMemPoolImportPointer",
    "cudaMallocFromPoolAsync",
    "cudaMallocFromPoolAsync_ptsz",
    "cudaSignalExternalSemaphoresAsync",
    "cudaSignalExternalSemaphoresAsync",
    "cudaWaitExternalSemaphoresAsync",
    "cudaWaitExternalSemaphoresAsync",
    "cudaGraphAddExternalSemaphoresSignalNode",
    "cudaGraphExternalSemaphoresSignalNodeGetParams",
    "cudaGraphExternalSemaphoresSignalNodeSetParams",
    "cudaGraphAddExternalSemaphoresWaitNode",
    "cudaGraphExternalSemaphoresWaitNodeGetParams",
    "cudaGraphExternalSemaphoresWaitNodeSetParams",
    "cudaGraphExecExternalSemaphoresSignalNodeSetParams",
    "cudaGraphExecExternalSemaphoresWaitNodeSetParams",
    "cudaDeviceFlushGPUDirectRDMAWrites",
    "cudaGetDriverEntryPoint",
    "cudaGetDriverEntryPoint_ptsz",
    "cudaGraphDebugDotPrint",
    "cudaStreamGetCaptureInfo_v2",
    "cudaStreamGetCaptureInfo_v2_ptsz",
    "cudaStreamUpdateCaptureDependencies",
    "cudaStreamUpdateCaptureDependencies_ptsz",
    "cudaUserObjectCreate",
    "cudaUserObjectRetain",
    "cudaUserObjectRelease",
    "cudaGraphRetainUserObject",
    "cudaGraphReleaseUserObject",
    "cudaGraphInstantiateWithFlags",
    "cudaGraphAddMemAllocNode",
    "cudaGraphMemAllocNodeGetParams",
    "cudaGraphAddMemFreeNode",
    "cudaGraphMemFreeNodeGetParams",
    "cudaDeviceGraphMemTrim",
    "cudaDeviceGetGraphMemAttribute",
    "cudaDeviceSetGraphMemAttribute",
    "cudaGraphNodeSetEnabled",
    "cudaGraphNodeGetEnabled",
    "cudaArrayGetMemoryRequirements",
    "cudaMipmappedArrayGetMemoryRequirements",
    "cudaLaunchKernelExC",
    "cudaLaunchKernelExC_ptsz",
    "cudaOccupancyMaxPotentialClusterSize",
    "cudaOccupancyMaxActiveClusters",
    "cudaCreateTextureObject_v2",
    "cudaGetTextureObjectTextureDesc_v2",
    "cudaGraphInstantiateWithParams",
    "cudaGraphInstantiateWithParams_ptsz",
    "cudaGraphExecGetFlags",
    "cuda439",
    "cudaGetDeviceProperties_v2",
    "cudaStreamGetId",
    "cudaStreamGetId_ptsz",
    "cudaGraphInstantiate",
    "cuda444",
    "SIZE"
};

const char* runtimeCbidName(CUpti_CallbackId cbid) {
  constexpr int names_size =
      sizeof(runtimeCbidNames) / sizeof(runtimeCbidNames[0]);
  if (cbid < 0 || cbid >= names_size) {
    return runtimeCbidNames[CUPTI_RUNTIME_TRACE_CBID_INVALID];
  }
  return runtimeCbidNames[cbid];
}

// From https://docs.nvidia.com/cupti/modules.html#group__CUPTI__ACTIVITY__API_1g80e1eb47615e31021f574df8ebbe5d9a
//   enum CUpti_ActivitySynchronizationType
const char* syncTypeString(
    CUpti_ActivitySynchronizationType kind) {
  switch (kind) {
    case CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_EVENT_SYNCHRONIZE:
      return "Event Sync";
    case CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT:
      return "Stream Wait Event";
    case CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_SYNCHRONIZE:
      return "Stream Sync";
    case CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_CONTEXT_SYNCHRONIZE:
      return "Context Sync";
    case CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_UNKNOWN:
    default:
      return "Unknown Sync";
  }
  return "<unknown>";
}
} // namespace libkineto
