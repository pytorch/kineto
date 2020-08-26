import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatCardModule} from '@angular/material/card';

import {RecommendationResultView} from './recommendation_result_view';

@NgModule({
  declarations: [RecommendationResultView],
  imports: [
    CommonModule,
    MatCardModule,
  ],
  exports: [RecommendationResultView]
})
export class RecommendationResultViewModule {
}
