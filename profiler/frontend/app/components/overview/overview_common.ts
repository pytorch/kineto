import {GeneralAnalysisOrNull, InputPipelineAnalysisOrNull, NormalizedAcceleratorPerformanceOrNull, OverviewDataTuple, RecommendationResultOrNull, RunEnvironmentOrNull} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {Diagnostics} from 'org_xprof/frontend/app/common/interfaces/diagnostics';
import {parseDiagnosticsDataTable} from 'org_xprof/frontend/app/common/utils/utils';

const GENERAL_ANALYSIS_INDEX = 0;
const INPUT_PIPELINE_ANALYSIS_INDEX = 1;
const RUN_ENVIRONMENT_INDEX = 2;
const RECOMMENDATION_RESULT_INDEX = 3;
const NORMALIZED_ACCELERATOR_PERFORMANCE_INDEX = 5;
const DIAGNOSTICS_INDEX = 6;

/** A common class of overview page component. */
export class OverviewCommon {
  private propertyValues: string[] = [];

  diagnostics: Diagnostics = {info: [], warnings: [], errors: []};
  generalAnalysis: GeneralAnalysisOrNull = null;
  inputPipelineAnalysis: InputPipelineAnalysisOrNull = null;
  recommendationResult: RecommendationResultOrNull = null;
  runEnvironment: RunEnvironmentOrNull = null;
  normalizedAcceleratorPerformance: NormalizedAcceleratorPerformanceOrNull =
      null;

  get averageStepTimePropertyValues(): string[] {
    return this.propertyValues;
  }

  set averageStepTimePropertyValues(propertyValues: string[]) {
    this.propertyValues = propertyValues;
  }

  parseOverviewData(data: OverviewDataTuple) {
    this.generalAnalysis = data[GENERAL_ANALYSIS_INDEX];
    this.inputPipelineAnalysis = data[INPUT_PIPELINE_ANALYSIS_INDEX];
    this.runEnvironment = data[RUN_ENVIRONMENT_INDEX];
    this.recommendationResult = data[RECOMMENDATION_RESULT_INDEX];
    this.normalizedAcceleratorPerformance =
        data[NORMALIZED_ACCELERATOR_PERFORMANCE_INDEX];
    this.diagnostics = parseDiagnosticsDataTable(data[DIAGNOSTICS_INDEX]);
  }
}
