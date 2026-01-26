"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

# Scalar hash (for reference/debugging)
    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        instrs = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            instrs.append({"alu": [
                (op1, tmp1, val_hash_addr, self.scratch_const(val1)),
                (op3, tmp2, val_hash_addr, self.scratch_const(val3)),
            ]})
            instrs.append({"alu": [(op2, val_hash_addr, tmp1, tmp2)]})
            instrs.append({"debug": [("compare", val_hash_addr, (round, i, "hash_stage", hi))]})
        return instrs

    # Vector hash - operates on VLEN (8) values at once
    # v_val, v_tmp1, v_tmp2, v_const1, v_const2 are base addresses of VLEN-sized scratch regions
    def build_hash_vec(self, v_val, v_tmp1, v_tmp2, v_const1, v_const2):
        instrs = []
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            # Broadcast scalar constants to vector registers
            const1_scalar = self.scratch_const(val1)
            const3_scalar = self.scratch_const(val3)
            instrs.append({"valu": [
                ("vbroadcast", v_const1, const1_scalar),
                ("vbroadcast", v_const2, const3_scalar),
            ]})
            # Pack two independent vector ALU ops
            instrs.append({"valu": [
                (op1, v_tmp1, v_val, v_const1),
                (op3, v_tmp2, v_val, v_const2),
            ]})
            instrs.append({"valu": [(op2, v_val, v_tmp1, v_tmp2)]})
        return instrs

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        SIMD implementation - processes VLEN (8) items at a time.
        Unrolled for maximum performance.
        """
        # Scalar temporaries
        tmp1 = self.alloc_scratch("tmp1")
        tmp_addr = self.alloc_scratch("tmp_addr")
        tmp_addr2 = self.alloc_scratch("tmp_addr2")

        # Scratch space addresses
        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)
        n_nodes_const = self.scratch_const(n_nodes)

        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting SIMD loop"))

        # Vector scratch registers (VLEN contiguous locations each)
        v_idx = self.alloc_scratch("v_idx", VLEN)
        v_val = self.alloc_scratch("v_val", VLEN)
        v_node_val = self.alloc_scratch("v_node_val", VLEN)
        v_tmp1 = self.alloc_scratch("v_tmp1", VLEN)
        v_tmp2 = self.alloc_scratch("v_tmp2", VLEN)
        v_tmp3 = self.alloc_scratch("v_tmp3", VLEN)
        v_const1 = self.alloc_scratch("v_const1", VLEN)
        v_const2 = self.alloc_scratch("v_const2", VLEN)
        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)

        # Scratch for 8 node_val addresses
        node_addrs = [self.alloc_scratch(f"node_addr_{j}") for j in range(VLEN)]

        # Pre-broadcast all hash constants to vector registers (do once)
        hash_v_consts = []
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            v_c1 = self.alloc_scratch(f"v_hash_c1_{val1}", VLEN)
            v_c2 = self.alloc_scratch(f"v_hash_c2_{val3}", VLEN)
            self.add("valu", ("vbroadcast", v_c1, self.scratch_const(val1)))
            self.add("valu", ("vbroadcast", v_c2, self.scratch_const(val3)))
            hash_v_consts.append((op1, v_c1, op2, op3, v_c2))

        # Broadcast other constants once
        self.add("valu", ("vbroadcast", v_zero, zero_const))
        self.add("valu", ("vbroadcast", v_one, one_const))
        self.add("valu", ("vbroadcast", v_two, two_const))
        self.add("valu", ("vbroadcast", v_n_nodes, n_nodes_const))

        body_instrs = []

        for rnd in range(rounds):
            for i in range(0, batch_size, VLEN):
                i_const = self.scratch_const(i)

                # Calculate base addresses for this batch of 8
                body_instrs.append({"alu": [
                    ("+", tmp_addr, self.scratch["inp_indices_p"], i_const),
                    ("+", tmp_addr2, self.scratch["inp_values_p"], i_const),
                ]})

                # vload 8 idx and 8 val values
                body_instrs.append({"load": [
                    ("vload", v_idx, tmp_addr),
                    ("vload", v_val, tmp_addr2),
                ]})

                # Load 8 node_val values (scattered - need individual loads)
                # Calculate all 8 addresses in ONE cycle (machine supports 12 ALU ops)
                body_instrs.append({"alu": [
                    ("+", node_addrs[j], self.scratch["forest_values_p"], v_idx + j)
                    for j in range(VLEN)
                ]})

                # Load 2 at a time (4 cycles total)
                for j in range(0, VLEN, 2):
                    body_instrs.append({"load": [
                        ("load", v_node_val + j, node_addrs[j]),
                        ("load", v_node_val + j + 1, node_addrs[j + 1]),
                    ]})

                # XOR: v_val = v_val ^ v_node_val
                body_instrs.append({"valu": [("^", v_val, v_val, v_node_val)]})

                # Vector hash using pre-broadcast constants
                for op1, v_c1, op2, op3, v_c2 in hash_v_consts:
                    body_instrs.append({"valu": [
                        (op1, v_tmp1, v_val, v_c1),
                        (op3, v_tmp2, v_val, v_c2),
                    ]})
                    body_instrs.append({"valu": [(op2, v_val, v_tmp1, v_tmp2)]})

                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                # Optimized: (val & 1) gives 0 for even, 1 for odd
                # So (val & 1) + 1 gives 1 for even, 2 for odd - exactly what we want!
                body_instrs.append({"valu": [
                    ("&", v_tmp1, v_val, v_one),   # tmp1 = val & 1 (0 or 1)
                    ("*", v_idx, v_idx, v_two),    # idx = idx * 2
                ]})
                body_instrs.append({"valu": [("+", v_tmp1, v_tmp1, v_one)]})  # tmp1 = tmp1 + 1 (1 or 2)
                body_instrs.append({"valu": [("+", v_idx, v_idx, v_tmp1)]})

                # idx = 0 if idx >= n_nodes else idx
                # Pack valu comparison with alu address calculation (different engines!)
                body_instrs.append({
                    "valu": [("<", v_tmp1, v_idx, v_n_nodes)],
                    "alu": [
                        ("+", tmp_addr, self.scratch["inp_indices_p"], i_const),
                        ("+", tmp_addr2, self.scratch["inp_values_p"], i_const),
                    ],
                })
                body_instrs.append({"flow": [("vselect", v_idx, v_tmp1, v_idx, v_zero)]})

                # Store results
                body_instrs.append({"store": [
                    ("vstore", tmp_addr, v_idx),
                    ("vstore", tmp_addr2, v_val),
                ]})

        self.instrs.extend(body_instrs)
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
