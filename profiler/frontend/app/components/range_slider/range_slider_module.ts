import {NgModule} from '@angular/core';
import {MatSliderModule} from '@angular/material/slider';

import {RangeSlider} from './range_slider';

/** A range slider component module. */
@NgModule({
  declarations: [RangeSlider],
  imports: [MatSliderModule],
  exports: [RangeSlider]
})
export class RangeSliderModule {
}
