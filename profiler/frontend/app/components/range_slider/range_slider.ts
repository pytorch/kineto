import {Component, ElementRef, EventEmitter, HostListener, Input, OnChanges, OnInit, Output, SimpleChanges, ViewChild} from '@angular/core';

/** The base interface for range value. */
export interface RangeValue {
  low: number;
  high: number;
}

const PADDING = 8;

/** A range slider component. */
@Component({
  selector: 'range-slider',
  templateUrl: './range_slider.ng.html',
  styleUrls: ['./range_slider.scss']
})
export class RangeSlider implements OnInit, OnChanges {
  /** The maximum value that the slider can have. */
  @Input() max = 100;

  /** The minimum value that the slider can have. */
  @Input() min = 0;

  /** The values at which the thumb will snap. */
  @Input() step = 1;

  /** Range value of the slider. */
  @Input() rangeValue: RangeValue = {low: 0, high: 100};

  /** Event emitted when the slider value has changed. */
  @Output() change = new EventEmitter<RangeValue>();

  @ViewChild('range', {static: true}) rangeRef!: ElementRef;

  @ViewChild('rangeBar', {static: true}) rangeBarRef!: ElementRef;

  firstSliderValue = 0;
  secondSliderValue = 100;

  ngOnInit() {
    this.onResize();
  }

  ngOnChanges(changes: SimpleChanges) {
    this.firstSliderValue = Math.min(this.rangeValue.low, this.rangeValue.high);
    this.secondSliderValue =
        Math.max(this.rangeValue.low, this.rangeValue.high);
    this.updateValues();
  }

  @HostListener('window:resize')
  onResize() {
    this.updateRangeBar();
  }

  updateFirstSliderValue(value: number) {
    this.firstSliderValue = value || 0;
    this.updateValues();
  }

  updateSecondSliderValue(value: number) {
    this.secondSliderValue = value || 0;
    this.updateValues();
  }

  updateValues() {
    this.rangeValue.low =
        Math.min(this.firstSliderValue, this.secondSliderValue);
    this.rangeValue.high =
        Math.max(this.firstSliderValue, this.secondSliderValue);
    this.updateRangeBar();
    this.change.emit(this.rangeValue);
  }

  updateRangeBar() {
    const rangeWidth = this.rangeRef.nativeElement.clientWidth - PADDING * 2;
    const left = Math.max(
                     0,
                     rangeWidth * (this.rangeValue.low - this.min) /
                         (this.max - this.min)) +
        PADDING;
    const right = Math.max(
                      0,
                      rangeWidth * (this.max - this.rangeValue.high) /
                          (this.max - this.min)) +
        PADDING;
    this.rangeBarRef.nativeElement.style.left = String(left) + 'px';
    this.rangeBarRef.nativeElement.style.right = String(right) + 'px';
  }
}
