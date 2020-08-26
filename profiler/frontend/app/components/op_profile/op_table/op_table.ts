import {Component, Input, OnDestroy} from '@angular/core';
import {Store} from '@ngrx/store';
import {Node} from 'org_xprof/frontend/app/common/interfaces/op_profile.proto';
import {setActiveOpProfileNodeAction} from 'org_xprof/frontend/app/store/actions';

/** An op table view component. */
@Component({
  selector: 'op-table',
  templateUrl: './op_table.ng.html',
  styleUrls: ['./op_table.scss']
})
export class OpTable implements OnDestroy {
  /** The root node. */
  @Input() rootNode?: Node;

  /** The property to sort by wasted time. */
  @Input() byWasted: boolean = false;

  /** The property to show top 90%. */
  @Input() showP90: boolean = false;

  /** The number of children nodes to be shown. */
  @Input() childrenCount: number = 10;

  selectedNode?: Node;

  constructor(private readonly store: Store<{}>) {}

  updateSelected(node?: Node) {
    this.selectedNode = node;
  }

  ngOnDestroy() {
    this.store.dispatch(
        setActiveOpProfileNodeAction({activeOpProfileNode: null}));
  }

  updateActive(node: Node|null) {
    this.store.dispatch(setActiveOpProfileNodeAction(
        {activeOpProfileNode: node || this.selectedNode || null}));
  }
}
