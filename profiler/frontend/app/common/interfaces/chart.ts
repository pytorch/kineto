import {SimpleDataTableOrNull} from './data_table';

/** The enumerator for Google Chart type. */
export const enum ChartType {
  UNKNOWN = '',
  AREA_CHART = 'AreaChart',
  BAR_CHART = 'BarChart',
  BUBBLE_CHART = 'BubbleChart',
  CANDLESTICK_CHART = 'CandlestickChart',
  COLUMN_CHART = 'ColumnChart',
  COMBO_CHART = 'ComboChart',
  HISTOGRAM = 'Histogram',
  LINE_CHART = 'LineChart',
  PIE_CHART = 'PieChart',
  SCATTER_CHART = 'ScatterChart',
  STEPPED_AREA_CHART = 'SteppedAreaChart',
  TABLE = 'Table',
}

/** The enumerator for data type. */
export const enum DataType {
  UNKNOWN = 0,
  ARRAY,
  DATA_TABLE,
}

/** The type for Google Chart class. */
export type ChartClass =
    google.visualization.AreaChart|google.visualization.BarChart|
    google.visualization.BubbleChart|
    google.visualization.CandlestickChart|
    google.visualization.ColumnChart|google.visualization.ComboChart|
    google.visualization.Histogram|google.visualization.LineChart|
    google.visualization.PieChart|google.visualization.ScatterChart|
    google.visualization.SteppedAreaChart|google.visualization.Table;

/** All chart options type. */
export type ChartOptions = google.visualization.AreaChartOptions|
                           google.visualization.BarChartOptions|
                           google.visualization.BubbleChartOptions|
                           google.visualization.CandlestickChartOptions|
                           google.visualization.ColumnChartOptions|
                           google.visualization.ComboChartOptions|
                           google.visualization.HistogramOptions|
                           google.visualization.LineChartOptions|
                           google.visualization.PieChartOptions|
                           google.visualization.ScatterChartOptions|
                           google.visualization.SteppedAreaChartOptions|
                           google.visualization.TableOptions;

/** The base interface for an information of chart data. */
export interface ChartDataInfo {
  data: SimpleDataTableOrNull;
  type: DataType;
  dataProvider: ChartDataProvider;
  sortColumns?: google.visualization.SortByColumn[];
  filters?: google.visualization.DataTableCellFilter[];
  options?: ChartOptions;
}

/** The base interface for a char data provider. */
export interface ChartDataProvider {
  setChart(chart: ChartClass): void;
  setData(dataInfo: ChartDataInfo): void;
  setFilters(filters: google.visualization.DataTableCellFilter[]): void;
  setSortColumns(sortColumns: google.visualization.SortByColumn[]): void;
  process(): google.visualization.DataTable|google.visualization.DataView|null;
  getOptions(): ChartOptions|null;
  setUpdateEventListener(callback: Function): void;
}
