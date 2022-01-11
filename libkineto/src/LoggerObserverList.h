// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#if !USE_GOOGLE_LOG

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "ILoggerObserver.h"

#include <memory>
#include <iostream>

namespace KINETO_NAMESPACE {

// This is a Lock-Free Implementation using shared_ptr and atomics, using refcount
// for garbage collection to address ABA problem, and atomic updates. This removes
// the need for mutex locks. All functions in this list will be atomic and safe.

class LoggerObserverList {
 private:
  struct Node {
    std::shared_ptr<ILoggerObserver> ptr;
    std::shared_ptr<Node> next;
  };
  // This is atomic, MUST ALWAYS use atomic_ functions for head.
  std::shared_ptr<Node> head{nullptr};

 public:
  LoggerObserverList()=default;
  ~LoggerObserverList()=default;

  class reference {
    // This is atomic, MUST ALWAYS use atomic_ functions for p.
    std::shared_ptr<Node> p;
   public:
    reference(std::shared_ptr<Node> p_) : p{p_} { }
    std::shared_ptr<ILoggerObserver>& operator* () { return  p->ptr; }
    std::shared_ptr<ILoggerObserver>* operator->() { return &p->ptr; }
  };

  auto front() const {
    return reference(atomic_load(&head));
  }

  void push_front(std::shared_ptr<ILoggerObserver>& ptr) {
    auto p = std::make_shared<Node>();
    p->ptr = ptr;
    p->next = atomic_load(&head);
    while (!atomic_compare_exchange_weak(&head, &p->next, p))
      { }
  }

  void pop_front() {
    auto p = atomic_load(&head);
    while (p && !atomic_compare_exchange_weak(&head, &p, p->next))
      { }
  }

  void remove(std::shared_ptr<ILoggerObserver>& ptr) {
    // Update head if head->ptr matches ptr.
    auto p = atomic_load(&head);
    while (p && p->ptr == ptr && !atomic_compare_exchange_weak(&head, &p, p->next))
      { }

    // Swing p->next if p->next->ptr matches ptr. Otherwise keep traversing.
    while (p && p->next) {
      auto expected = atomic_load(&p->next);
      while (p && p->next && p->next->ptr == ptr &&
             !atomic_compare_exchange_weak(&p->next, &expected, p->next->next))
        { }
      p = atomic_load(&p->next);
    }
  }

  void write_all(const std::string& message, LoggerOutputType otype) {
    auto p = atomic_load(&head);
    while (p) {
      p->ptr->write(message, otype);
      p = atomic_load(&p->next);
    }
  }

  LoggerObserverList(LoggerObserverList&)=delete;
  void operator=(LoggerObserverList&)=delete;

};

} // namespace KINETO_NAMESPACE

#endif // !USE_GOOGLE_LOG
