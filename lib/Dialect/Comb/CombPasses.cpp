#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace comb {
namespace minimizer {

// A cube should be a multivalued expression.
// However, now we only use it as binary values.
using Var = llvm::BitVector;
using Cube = llvm::SmallVector<Var, 0>;

Cube from(llvm::StringRef string) {
  Cube cube;
  cube.reserve(string.size());
  for (auto ch : string) {
    Var var(2);
    switch (ch) {
    case '-':
      var.set(1);
    case '0':
      var.set(0);
      break;
    case '1':
      var.set(1);
      break;
    }
    cube.push_back(var);
  }
  return cube;
}

std::string to(Cube cube) {
  std::string str;
  str.reserve(cube.size());
  for (auto var : cube) {
    assert(var.size() == 2);
    bool b0 = var.test(0), b1 = var.test(1);
    if (b0 && b1)
      str.push_back('-');
    else if (b0)
      str.push_back('0');
    else
      str.push_back('1');
  }
  return str;
}

llvm::SmallVector<Cube> minimize(llvm::ArrayRef<Cube> on) {
  llvm::for_each(on, [](Cube cube) { llvm::outs() << to(cube) << "\n"; });
  return {};
}

} // namespace minimizer

#define GEN_PASS_CLASSES
#include "circt/Dialect/Comb/CombPasses.h.inc"

struct Minimization : public MinimizationBase<Minimization> {
  void runOnOperation() override {
    using namespace minimizer;
    mlir::OperationPass<mlir::ModuleOp>::getOperation()->walk(
        [](MinimizeOp op) {
          auto table = op->getAttrOfType<mlir::ArrayAttr>("table");

          llvm::SmallVector<Cube> cubes;

          llvm::transform(
              table, std::back_inserter(cubes), [](mlir::Attribute str) {
                return from(str.template cast<mlir::StringAttr>().strref());
              });

          minimize(cubes);

          //          auto table = ArrayAttr::get(
          //              op.getContext(), {
          //                                   StringAttr::get(op.getContext(),
          //                                   "test1"),
          //                                   StringAttr::get(op.getContext(),
          //                                   "test2"),
          //                                   StringAttr::get(op.getContext(),
          //                                   "test3"),
          //                                   StringAttr::get(op.getContext(),
          //                                   "test4"),
          //                               });
          op->setAttr("table", table);
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
