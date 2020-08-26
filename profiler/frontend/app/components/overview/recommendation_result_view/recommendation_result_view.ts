import {Component} from '@angular/core';
import {Store} from '@ngrx/store';

import {addAnchorTag, convertKnownToolToAnchorTag} from 'org_xprof/frontend/app/common/utils/utils';
import {setCurrentToolStateAction} from 'org_xprof/frontend/app/store/actions';

import {RecommendationResultViewCommon} from './recommendation_result_view_common';

const STATEMENT_INFO = [
  {id: 'eager_statement_html'},
  {id: 'tf_function_statement_html'},
  {id: 'statement'},
  {id: 'kernel_launch_statement'},
  {id: 'all_other_statement'},
  {id: 'precision_statement'},
];

/** A recommendation result view component. */
@Component({
  selector: 'recommendation-result-view',
  templateUrl: './recommendation_result_view.ng.html',
  styleUrls: ['./recommendation_result_view.scss']
})
export class RecommendationResultView extends RecommendationResultViewCommon {
  constructor(private readonly store: Store<{}>) {
    super();
  }

  getRecommendationResultProp(id: string, defaultValue: string = ''): string {
    const props = (this.recommendationResult || {}).p || {};

    switch (id) {
      case 'bottleneck':
        return props.bottleneck || defaultValue;
      case 'eager_statement_html':
        return props.eager_statement_html || defaultValue;
      case 'tf_function_statement_html':
        return props.tf_function_statement_html || defaultValue;
      case 'statement':
        return props.statement || defaultValue;
      case 'kernel_launch_statement':
        return props.kernel_launch_statement || defaultValue;
      case 'all_other_statement':
        return props.all_other_statement || defaultValue;
      case 'precision_statement':
        return props.precision_statement || defaultValue;
      default:
        break;
    }

    return defaultValue;
  }

  parseStatements() {
    this.statements = [];
    STATEMENT_INFO.forEach(info => {
      const prop = this.getRecommendationResultProp(info.id);
      if (prop) {
        this.statements.push({value: prop});
      }
    });
  }

  parseTips() {
    const data = this.recommendationResult || {};
    const hostTips: string[] = [];
    const deviceTips: string[] = [];
    const documentationTips: string[] = [];
    const faqTips: string[] = [];
    data.rows = data.rows || [];
    data.rows.forEach(row => {
      if (row.c && row.c[0] && row.c[0].v && row.c[1] && row.c[1].v) {
        switch (row.c[0].v) {
          case 'host':
            hostTips.push(convertKnownToolToAnchorTag(String(row.c[1].v)));
            break;
          case 'device':
            deviceTips.push(convertKnownToolToAnchorTag(String(row.c[1].v)));
            break;
          case 'doc':
            documentationTips.push(String(row.c[1].v));
            break;
          case 'faq':
            faqTips.push(String(row.c[1].v));
            break;
          default:
            break;
        }
      }
    });
    const bottleneck = this.getRecommendationResultProp('bottleneck');
    if (bottleneck === 'device') {
      hostTips.length = 0;
    } else if (bottleneck === 'host') {
      deviceTips.length = 0;
    }

    const tipInfoArray = [
      {
        title: 'Tool troubleshooting / FAQ',
        tips: faqTips,
      },
      {
        title: 'Next tools to use for reducing the input time',
        tips: hostTips,
        useClickCallback: true,
      },
      {
        title: 'Next tools to use for reducing the Device time',
        tips: deviceTips,
        useClickCallback: true,
      },
      {
        title: 'Other useful resources',
        tips: documentationTips,
      },
    ];
    this.tipInfoArray = tipInfoArray.filter(tipInfo => tipInfo.tips.length > 0);
  }

  onTipsClick(event: Event) {
    if (!event || !event.target ||
        (event.target as HTMLElement).tagName !== 'A') {
      return;
    }
    const tool = (event.target as HTMLElement).innerText;
    if (convertKnownToolToAnchorTag(tool) === addAnchorTag(tool)) {
      this.store.dispatch(setCurrentToolStateAction({currentTool: tool}));
    }
  }
}
