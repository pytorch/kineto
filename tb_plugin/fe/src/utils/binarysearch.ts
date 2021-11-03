export function binarySearch(
  arr: Array<any>,
  key: any,
  compare_fn: Function
): number {
  let low = 0,
    high = arr.length - 1
  while (low <= high) {
    let mid = Math.round((high + low) / 2)
    let cmp = compare_fn(key, arr[mid])
    if (cmp > 0) {
      low = mid + 1
    } else if (cmp < 0) {
      high = mid - 1
    } else {
      return mid
    }
  }
  return -1
}
