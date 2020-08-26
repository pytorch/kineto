import {NgModule} from '@angular/core';
import {MatCardModule} from '@angular/material/card';

import {BufferDetails} from './buffer_details';

/** A buffer details view module. */
@NgModule({
  declarations: [BufferDetails],
  imports: [MatCardModule],
  exports: [BufferDetails]
})
export class BufferDetailsModule {
}
