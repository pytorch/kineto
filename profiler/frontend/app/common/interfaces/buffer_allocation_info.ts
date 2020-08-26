/** The base interface for a buffer allocation information. */
export interface BufferAllocationInfo {
  size: number;
  alloc: number;
  free: number;
  color?: string;
}
