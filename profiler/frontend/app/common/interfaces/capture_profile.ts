/** The base interface for a response of capture profile api. */
export interface CaptureProfileResponse {
  error?: string;
  result?: string;
}

/** The base interface for a options of capture profile api. */
export interface CaptureProfileOptions {
  serviceAddr: string;
  isTpuName: boolean;
  duration: number;
  numRetry: number;
  workerList: string;
  hostTracerLevel: number;
  deviceTracerLevel: number;
  pythonTracerLevel: number;
}
