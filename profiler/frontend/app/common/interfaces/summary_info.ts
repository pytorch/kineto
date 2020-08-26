/** The base interface for a information of performance summary view. */
export interface SummaryInfo {
  type?: string;
  title: string;
  descriptions?: string[];
  tooltip?: string;
  value?: string;
  valueColor?: string;
  propertyValues?: string[];
}
