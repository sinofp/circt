#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace comb {

#define GEN_PASS_CLASSES
#include "circt/Dialect/Comb/CombPasses.h.inc"

struct Minimization : public MinimizationBase<Minimization> {
  void runOnOperation() override {
    mlir::OperationPass<mlir::ModuleOp>::getOperation()->walk(
        [](MinimizeOp op) {
          auto table = ArrayAttr::get(
              op.getContext(), {
                                   StringAttr::get(op.getContext(), "test1"),
                                   StringAttr::get(op.getContext(), "test2"),
                                   StringAttr::get(op.getContext(), "test3"),
                                   StringAttr::get(op.getContext(), "test4"),
                               });
          op->setAttr("truthTable", table);
        });
  }
};

std::unique_ptr<mlir::Pass> createMinimizationPass() {
  return std::make_unique<Minimization>();
}

#define GEN_PASS_REGISTRATION
#include "circt/Dialect/Comb/CombPasses.h.inc"
void registerCombPasses() { registerPasses(); }

} // namespace comb
} // namespace circt
