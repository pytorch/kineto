import {Input, OnChanges, SimpleChanges} from '@angular/core';

import {RecommendationResult} from 'org_xprof/frontend/app/common/interfaces/data_table';

interface StatementData {
  value: string;
  color?: string;
}
interface TipInfo {
  title: string;
  style?: {[key: string]: string};
  tips: string[];
  useClickCallback?: boolean;
}

/** A common class of recommendation result view component. */
export class RecommendationResultViewCommon implements OnChanges {
  /** The recommendation result data. */
  @Input() recommendationResult?: RecommendationResult;

  title = 'Recommendation for Next Step';
  statements: StatementData[] = [];
  tipInfoArray: TipInfo[] = [];

  ngOnChanges(changes: SimpleChanges) {
    this.parseStatements();
    this.parseTips();
  }

  parseStatements() {}

  parseTips() {}

  onTipsClick(event: Event) {}
}
