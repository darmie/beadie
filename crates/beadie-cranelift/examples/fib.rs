// fib.rs — Full pipeline: source → AST → MIR → interpreter → beadie → Cranelift JIT.
//
// Demonstrates the three compilation phases of a language runtime:
//
//   Phase 1: Frontend — parse source into an AST
//   Phase 2: Lowering — compile AST into register-based MIR (bytecode)
//   Phase 3: JIT      — beadie promotes hot MIR to native code via Cranelift
//
// The interpreter runs MIR. Beadie sits on the dispatch path and
// automatically promotes functions that cross the hotness threshold.
// queue_ahead lets you tune for backend compile latency so native code
// is ready by the time the function is truly hot.
//
// Run:
//   cargo run -p beadie-cranelift --example fib
//
// Enable beadie trace logs:
//   RUST_LOG=beadie_core=debug cargo run -p beadie-cranelift --example fib

use std::ptr;
use std::sync::Arc;
use std::time::Instant;

use beadie_backend::{BackendAdapter, BoundBead};
use beadie_core::ThresholdPolicy;
use beadie_cranelift::{CraneliftBackend, CraneliftConfig, CraneliftFunctionDef};
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{types, AbiParam, InstBuilder};

// ─────────────────────────────────────────────────────────────────────────────
// Phase 1: Frontend — source language AST
// ─────────────────────────────────────────────────────────────────────────────

/// Expression AST — what a parser produces from source code.
enum Expr {
    /// Integer literal
    Lit(i64),
    /// Function argument (by index)
    Arg,
    /// Binary operation
    Bin(BinOp, Box<Expr>, Box<Expr>),
    /// Call function by id: call(func_id, arg)
    Call(usize, Box<Expr>),
    /// if cond then else
    If(Box<Expr>, Box<Expr>, Box<Expr>),
}

#[derive(Clone, Copy)]
enum BinOp {
    Add,
    Sub,
    Lt,
}

/// A parsed function definition.
struct FuncDef {
    name: String,
    body: Expr,
}

/// "Parse" the fibonacci function.
///
/// A real frontend would tokenize + parse source text into this AST.
/// For this example we construct it directly — the point is to show
/// that MIR is *compiled output*, not the starting representation.
///
/// Source equivalent:
/// ```text
/// fn fib(n) = if n < 2 then n else fib(n - 1) + fib(n - 2)
/// ```
fn parse_fib() -> FuncDef {
    use BinOp::*;
    use Expr::*;
    FuncDef {
        name: "fib".into(),
        body: If(
            Box::new(Bin(Lt, Box::new(Arg), Box::new(Lit(2)))),
            Box::new(Arg),
            Box::new(Bin(
                Add,
                Box::new(Call(0, Box::new(Bin(Sub, Box::new(Arg), Box::new(Lit(1)))))),
                Box::new(Call(0, Box::new(Bin(Sub, Box::new(Arg), Box::new(Lit(2)))))),
            )),
        ),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 2: Lowering — AST → register-based MIR
// ─────────────────────────────────────────────────────────────────────────────

type Reg = usize;

#[derive(Clone)]
enum Inst {
    Iconst(Reg, i64),
    #[allow(dead_code)]
    Arg(Reg, usize),
    Add(Reg, Reg, Reg),
    Sub(Reg, Reg, Reg),
    Lt(Reg, Reg, Reg),
    Call(Reg, usize, Reg),
    Ret(Reg),
    BrIf(Reg, usize, usize),
}

#[derive(Clone)]
struct MirFunc {
    name: String,
    num_regs: usize,
    blocks: Vec<Vec<Inst>>,
}

/// Compiler: lowers an AST expression into register-based MIR.
struct Lowering {
    blocks: Vec<Vec<Inst>>,
    next_reg: usize,
}

impl Lowering {
    fn new() -> Self {
        Self {
            blocks: vec![Vec::new()],
            next_reg: 0,
        }
    }

    fn alloc(&mut self) -> Reg {
        let r = self.next_reg;
        self.next_reg += 1;
        r
    }

    fn emit(&mut self, block: usize, inst: Inst) {
        self.blocks[block].push(inst);
    }

    fn new_block(&mut self) -> usize {
        let id = self.blocks.len();
        self.blocks.push(Vec::new());
        id
    }

    /// Lower an expression into MIR, returning the register holding the result.
    fn lower(&mut self, block: usize, expr: &Expr) -> Reg {
        match expr {
            Expr::Lit(v) => {
                let r = self.alloc();
                self.emit(block, Inst::Iconst(r, *v));
                r
            }
            Expr::Arg => {
                let r = self.alloc();
                self.emit(block, Inst::Arg(r, 0));
                r
            }
            Expr::Bin(op, lhs, rhs) => {
                let a = self.lower(block, lhs);
                let b = self.lower(block, rhs);
                let r = self.alloc();
                self.emit(
                    block,
                    match op {
                        BinOp::Add => Inst::Add(r, a, b),
                        BinOp::Sub => Inst::Sub(r, a, b),
                        BinOp::Lt => Inst::Lt(r, a, b),
                    },
                );
                r
            }
            Expr::Call(fid, arg) => {
                let a = self.lower(block, arg);
                let r = self.alloc();
                self.emit(block, Inst::Call(r, *fid, a));
                r
            }
            Expr::If(cond, then_e, else_e) => {
                let c = self.lower(block, cond);
                let then_b = self.new_block();
                let else_b = self.new_block();
                self.emit(block, Inst::BrIf(c, then_b, else_b));

                let tr = self.lower(then_b, then_e);
                self.emit(then_b, Inst::Ret(tr));

                let er = self.lower(else_b, else_e);
                self.emit(else_b, Inst::Ret(er));

                0 // both branches return — caller never uses this
            }
        }
    }
}

/// Compile a function definition from AST to MIR.
fn compile(def: &FuncDef) -> MirFunc {
    let mut low = Lowering::new();

    // If the top-level expression is an If (both branches return),
    // the lowering handles it. Otherwise wrap in a Ret.
    match &def.body {
        Expr::If(..) => {
            low.lower(0, &def.body);
        }
        other => {
            let r = low.lower(0, other);
            low.emit(0, Inst::Ret(r));
        }
    }

    MirFunc {
        name: def.name.clone(),
        num_regs: low.next_reg,
        blocks: low.blocks,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 3: Runtime — interpreter + beadie dispatch via BackendAdapter
// ─────────────────────────────────────────────────────────────────────────────

struct Runtime {
    adapter: BackendAdapter<CraneliftBackend>,
    functions: Vec<Arc<MirFunc>>,
    beads: Vec<BoundBead<CraneliftBackend>>,
}

/// Trampoline callable from JIT'd code. Compiled functions call this for
/// every function invocation, routing back through beadie's dispatch.
extern "C" fn dispatch_trampoline(runtime: *const u8, func_id: i64, arg: i64) -> i64 {
    let rt = unsafe { &*(runtime as *const Runtime) };
    rt.call(func_id as usize, arg)
}

impl Runtime {
    fn new(functions: Vec<MirFunc>, threshold: u32, queue_ahead: u32) -> Self {
        let backend = CraneliftConfig::new()
            .opt_level("speed")
            .symbol("__beadie_dispatch", dispatch_trampoline as *const u8)
            .build()
            .expect("cranelift backend");

        // queue_ahead: submit compile job early to absorb backend latency.
        // Tune to your backend — Cranelift ~5ms, LLVM ~50ms.
        let policy = ThresholdPolicy::new(threshold).queue_ahead(queue_ahead);
        let adapter = BackendAdapter::with_policy(backend, policy);

        let beads: Vec<_> = functions
            .iter()
            .map(|_| adapter.register(ptr::null_mut(), None))
            .collect();

        Self {
            adapter,
            functions: functions.into_iter().map(Arc::new).collect(),
            beads,
        }
    }

    fn call(&self, func_id: usize, arg: i64) -> i64 {
        let bound = &self.beads[func_id];
        let mir = Arc::clone(&self.functions[func_id]);
        let backend = Arc::clone(bound.backend());

        if let Some(code) = self
            .adapter
            .on_invoke(bound, move |bead| build_def(&mir, &backend, bead))
        {
            let f: extern "C" fn(*const u8, i64) -> i64 = unsafe { std::mem::transmute(code) };
            f(self as *const Self as *const u8, arg)
        } else {
            self.interpret(func_id, arg)
        }
    }

    fn interpret(&self, func_id: usize, arg: i64) -> i64 {
        let func = &self.functions[func_id];
        let mut regs = vec![0i64; func.num_regs];
        let mut block = 0usize;

        loop {
            for inst in &func.blocks[block] {
                match *inst {
                    Inst::Iconst(d, v) => regs[d] = v,
                    Inst::Arg(d, _) => regs[d] = arg,
                    Inst::Add(d, a, b) => regs[d] = regs[a].wrapping_add(regs[b]),
                    Inst::Sub(d, a, b) => regs[d] = regs[a].wrapping_sub(regs[b]),
                    Inst::Lt(d, a, b) => regs[d] = (regs[a] < regs[b]) as i64,
                    Inst::Call(d, fid, a) => regs[d] = self.call(fid, regs[a]),
                    Inst::Ret(s) => return regs[s],
                    Inst::BrIf(c, t, e) => {
                        block = if regs[c] != 0 { t } else { e };
                        break;
                    }
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// JIT backend — MIR → CraneliftFunctionDef for the BackendAdapter
// ─────────────────────────────────────────────────────────────────────────────

/// Build a [`CraneliftFunctionDef`] from MIR. Called by the broker thread
/// when a function wins promotion. The adapter passes the def to
/// [`CraneliftBackend::compile`].
fn build_def(
    mir: &MirFunc,
    backend: &CraneliftBackend,
    _bead: &Arc<beadie_core::Bead>,
) -> CraneliftFunctionDef {
    // compiled function: (runtime_ptr, arg) -> result
    let mut fn_sig = backend.make_signature();
    fn_sig.params.push(AbiParam::new(types::I64));
    fn_sig.params.push(AbiParam::new(types::I64));
    fn_sig.returns.push(AbiParam::new(types::I64));

    // dispatch trampoline: (runtime_ptr, func_id, arg) -> result
    let mut dispatch_sig = backend.make_signature();
    dispatch_sig.params.push(AbiParam::new(types::I64));
    dispatch_sig.params.push(AbiParam::new(types::I64));
    dispatch_sig.params.push(AbiParam::new(types::I64));
    dispatch_sig.returns.push(AbiParam::new(types::I64));

    let name = format!("__beadie_{}", mir.name);
    let mut def = backend.new_def(fn_sig, &name).unwrap();

    let dispatch_ref = backend
        .import_function("__beadie_dispatch", &dispatch_sig, &mut def.ctx.func)
        .unwrap();

    {
        let mut b = def.builder();

        let blocks: Vec<_> = mir.blocks.iter().map(|_| b.create_block()).collect();

        let vars: Vec<_> = (0..=mir.num_regs)
            .map(|_| b.declare_var(types::I64))
            .collect();
        let runtime_var = vars[mir.num_regs];

        b.append_block_params_for_function_params(blocks[0]);
        b.switch_to_block(blocks[0]);
        let runtime_val = b.block_params(blocks[0])[0];
        let arg_val = b.block_params(blocks[0])[1];
        b.def_var(runtime_var, runtime_val);

        for (bi, mir_block) in mir.blocks.iter().enumerate() {
            if bi > 0 {
                b.switch_to_block(blocks[bi]);
            }
            for inst in mir_block {
                match *inst {
                    Inst::Iconst(d, val) => {
                        let v = b.ins().iconst(types::I64, val);
                        b.def_var(vars[d], v);
                    }
                    Inst::Arg(d, _) => {
                        b.def_var(vars[d], arg_val);
                    }
                    Inst::Add(d, a, rb) => {
                        let va = b.use_var(vars[a]);
                        let vb = b.use_var(vars[rb]);
                        let v = b.ins().iadd(va, vb);
                        b.def_var(vars[d], v);
                    }
                    Inst::Sub(d, a, rb) => {
                        let va = b.use_var(vars[a]);
                        let vb = b.use_var(vars[rb]);
                        let v = b.ins().isub(va, vb);
                        b.def_var(vars[d], v);
                    }
                    Inst::Lt(d, a, rb) => {
                        let va = b.use_var(vars[a]);
                        let vb = b.use_var(vars[rb]);
                        let cmp = b.ins().icmp(IntCC::SignedLessThan, va, vb);
                        let v = b.ins().uextend(types::I64, cmp);
                        b.def_var(vars[d], v);
                    }
                    Inst::Call(d, fid, arg_reg) => {
                        let rt = b.use_var(runtime_var);
                        let fid_val = b.ins().iconst(types::I64, fid as i64);
                        let a = b.use_var(vars[arg_reg]);
                        let call = b.ins().call(dispatch_ref, &[rt, fid_val, a]);
                        let result = b.inst_results(call)[0];
                        b.def_var(vars[d], result);
                    }
                    Inst::Ret(s) => {
                        let v = b.use_var(vars[s]);
                        b.ins().return_(&[v]);
                    }
                    Inst::BrIf(c, then_b, else_b) => {
                        let cond = b.use_var(vars[c]);
                        let cmp = b.ins().icmp_imm(IntCC::NotEqual, cond, 0);
                        b.ins().brif(cmp, blocks[then_b], &[], blocks[else_b], &[]);
                    }
                }
            }
        }

        b.seal_all_blocks();
        b.finalize();
    }

    def
}

// ─────────────────────────────────────────────────────────────────────────────
// Baselines — isolate beadie's overhead
// ─────────────────────────────────────────────────────────────────────────────

/// Pure interpreter — no beadie, no dispatch check. Measures raw MIR
/// interpretation cost.
fn interpret_raw(func: &MirFunc, arg: i64) -> i64 {
    let mut regs = vec![0i64; func.num_regs];
    let mut block = 0usize;
    loop {
        for inst in &func.blocks[block] {
            match *inst {
                Inst::Iconst(d, v) => regs[d] = v,
                Inst::Arg(d, _) => regs[d] = arg,
                Inst::Add(d, a, b) => regs[d] = regs[a].wrapping_add(regs[b]),
                Inst::Sub(d, a, b) => regs[d] = regs[a].wrapping_sub(regs[b]),
                Inst::Lt(d, a, b) => regs[d] = (regs[a] < regs[b]) as i64,
                Inst::Call(d, _, a) => regs[d] = interpret_raw(func, regs[a]),
                Inst::Ret(s) => return regs[s],
                Inst::BrIf(c, t, e) => {
                    block = if regs[c] != 0 { t } else { e };
                    break;
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// main — three-phase pipeline
// ─────────────────────────────────────────────────────────────────────────────

/// fib(20) produces 21,891 recursive dispatches per call.
const DISPATCHES_PER_CALL: u64 = 21_891;

fn print_row(label: &str, elapsed: std::time::Duration, calls: u64) {
    let total = calls * DISPATCHES_PER_CALL;
    let throughput = total as f64 / elapsed.as_secs_f64() / 1_000_000.0;
    println!("  {label:<24} {elapsed:>8.1?}   {throughput:>6.1}M dispatch/s");
}

fn run_beadie(label: &str, program: &[MirFunc], threshold: u32, queue_ahead: u32, calls: u64) {
    let rt = Runtime::new(program.to_vec(), threshold, queue_ahead);

    let t = Instant::now();
    for _ in 0..calls {
        rt.call(0, 20);
    }
    let elapsed = t.elapsed();
    print_row(label, elapsed, calls);
}

fn main() {
    env_logger::init();

    // ── Phase 1: Frontend ────────────────────────────────────────────────────
    println!("=== Phase 1: Frontend (source -> AST) ===");
    let t = Instant::now();
    let ast = parse_fib();
    println!("  parsed '{}' [{:.1?}]\n", ast.name, t.elapsed());

    // ── Phase 2: Lowering ────────────────────────────────────────────────────
    println!("=== Phase 2: Lowering (AST -> MIR) ===");
    let t = Instant::now();
    let mir = compile(&ast);
    println!(
        "  compiled '{}' -> {} regs, {} blocks [{:.1?}]\n",
        mir.name,
        mir.num_regs,
        mir.blocks.len(),
        t.elapsed(),
    );

    // ── Phase 3: Execution ───────────────────────────────────────────────────
    let program = vec![mir.clone()];
    let calls: u64 = 500;

    println!("=== Phase 3: Execution (fib(20) x {calls}) ===");
    println!("  {:<24} {:>8}   {:>15}", "config", "time", "throughput");
    println!("  {:-<24} {:-<8}   {:-<15}", "", "", "");

    // Baseline: pure interpreter, no beadie overhead.
    {
        let t = Instant::now();
        for _ in 0..calls {
            std::hint::black_box(interpret_raw(&mir, 20));
        }
        print_row("interpreter (no beadie)", t.elapsed(), calls);
    }

    // Beadie: interpreter → JIT with varying queue_ahead.
    run_beadie("beadie (qa=0)", &program, 1000, 0, calls);
    run_beadie("beadie (qa=500)", &program, 1000, 500, calls);
    run_beadie("beadie (qa=990)", &program, 1000, 990, calls);

    // Baseline: pre-compiled, measures dispatch trampoline + beadie fast path.
    // threshold=1 + queue_ahead=1 forces immediate compilation on first call.
    run_beadie("beadie (pre-compiled)", &program, 1, 1, calls);

    println!();
}
