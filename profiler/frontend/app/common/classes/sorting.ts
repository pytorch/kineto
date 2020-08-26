function consumeNumber(s: string, i: number): number {
  const State = {NATURAL: 0, REAL: 1, EXPONENT_SIGN: 2, EXPONENT: 3};
  let state = State.NATURAL;
  for (; i < s.length; i++) {
    if (state === State.NATURAL) {
      if (s[i] === '.') {
        state = State.REAL;
      } else if (s[i] === 'e' || s[i] === 'E') {
        state = State.EXPONENT_SIGN;
      } else if (!isDigit(s[i])) {
        break;
      }
    } else if (state === State.REAL) {
      if (s[i] === 'e' || s[i] === 'E') {
        state = State.EXPONENT_SIGN;
      } else if (!isDigit(s[i])) {
        break;
      }
    } else if (state === State.EXPONENT_SIGN) {
      if (isDigit(s[i]) || s[i] === '+' || s[i] === '-') {
        state = State.EXPONENT;
      } else {
        break;
      }
    } else if (state === State.EXPONENT) {
      if (!isDigit(s[i])) {
        break;
      }
    }
  }
  return i;
}

function isDigit(c: string) {
  return '0' <= c && c <= '9';
}

function isBreak(c: string) {
  return c === '/' || c === '_' || isDigit(c);
}

/**
 * Compares tag names asciinumerically broken into components.
 *
 * This is the comparison function used for sorting most string values in
 * TensorBoard. Unlike the standard asciibetical comparator, this function
 * knows that 'a10b' > 'a2b'. Fixed point and engineering notation are
 * supported. This function also splits the input by slash and underscore to
 * perform array comparison. Therefore it knows that 'a/a' < 'a+/a' even
 * though '+' < '/' in the ASCII table.
 */
export function compareTagNames(a: string, b: string): number {
  let ai = 0;
  let bi = 0;
  while (true) {
    if (ai === a.length) {
      return bi === b.length ? 0 : -1;
    }
    if (bi === b.length) {
      return 1;
    }
    if (isDigit(a[ai]) && isDigit(b[bi])) {
      const ais = ai;
      const bis = bi;
      ai = consumeNumber(a, ai + 1);
      bi = consumeNumber(b, bi + 1);
      const an = parseFloat(a.slice(ais, ai));
      const bn = parseFloat(b.slice(bis, bi));
      if (an < bn) {
        return -1;
      }
      if (an > bn) {
        return 1;
      }
      continue;
    }
    if (isBreak(a[ai])) {
      if (!isBreak(b[bi])) {
        return -1;
      }
    } else if (isBreak(b[bi])) {
      return 1;
    } else if (a[ai] < b[bi]) {
      return -1;
    } else if (a[ai] > b[bi]) {
      return 1;
    }
    ai++;
    bi++;
  }
}
