import {Component, Input} from '@angular/core';

import {InputPipelineDeviceAnalysis} from 'org_xprof/frontend/app/common/interfaces/data_table';

/** A summary of input pipeline analysis component. */
@Component({
  selector: 'analysis-summary',
  templateUrl: './analysis_summary.ng.html',
  styleUrls: ['./analysis_summary.scss']
})
export class AnalysisSummary {
  /** The input pipeline device anaysis data. */
  @Input()
  set deviceAnalysis(analysis: InputPipelineDeviceAnalysis|null) {
    analysis = analysis || {};
    analysis.p = analysis.p || {};
    this.inputConclusion = analysis.p.input_conclusion || '';
    this.summaryNextstep =
      this.replaceSectionName(analysis.p.summary_nextstep || '');
    this.summaryColor = 'green';
    if (this.inputConclusion.includes('HIGHLY')) {
      this.summaryColor = 'red';
    } else if (this.inputConclusion.includes('MODERATE')) {
      this.summaryColor = 'orange';
    }
  }

  inputConclusion = '';
  summaryNextstep = '';
  summaryColor = 'green';

  private replaceSectionName(summary: string): string {
    return summary.replace(/section 2/g, 'Device-side analysis details section')
        .replace(/section 3/g, 'Host-side analysis details section');
  }
}
