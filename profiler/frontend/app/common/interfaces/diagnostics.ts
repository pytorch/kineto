/** The base interface for diagnosing profiling issues. */
export interface Diagnostics {
  info: string[];
  errors: string[];
  warnings: string[];
}
