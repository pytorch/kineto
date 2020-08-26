# Range Slider
<range-slider> allows for the selection of a range from a slider via mouse, touch, or keyboard.

## Selecting a range
By default the minimum value of the slider is 0, the maximum value is 100, and the thumb moves in increments of 1. These values can be changed by setting the `min`, `max`, and `step` attributes respectively. The initial range's row is set to the minimum value and range's high is set to the maximum value unless otherwise specifiled.
```
<range-slider min="1" max="5" step="0.5" [rangeValue]="{low: 2.5, high: 3.5}"></range-slider>
```

## Handling an event
Whenever the range value changes, a change event is emitted. Events have low and high values.
```
<range-slider (change)="handleEvent($event.low, $event.high)"></range-slider>

handleEvent(low: number, high: number) {...}
```
