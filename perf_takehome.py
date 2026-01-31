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
        Triple-buffered SIMD implementation with 3-batch parallel processing.

        Key optimization: Process 3 batches in parallel to utilize all 6 VALU slots.
        - Each hash stage uses 2 VALU ops per batch
        - 3 batches × 2 ops = 6 ops (full utilization!)

        Pipeline structure:
        - Prologue: Load first 3 batches into A, B, C
        - Steady state: Process 3 batches at a time
            - Compute A+B+C in parallel while loading next batch
            - Store A+B+C, load more
        - Epilogue: Handle remaining batches
        """
        # Scalar temporaries
        tmp1 = self.alloc_scratch("tmp1")

        # Address registers for each buffer
        addr_idx_A = self.alloc_scratch("addr_idx_A")
        addr_val_A = self.alloc_scratch("addr_val_A")
        addr_idx_B = self.alloc_scratch("addr_idx_B")
        addr_val_B = self.alloc_scratch("addr_val_B")
        addr_idx_C = self.alloc_scratch("addr_idx_C")
        addr_val_C = self.alloc_scratch("addr_val_C")

        # Scratch space addresses
        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for idx, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, idx))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)
        n_nodes_const = self.scratch_const(n_nodes)

        self.add("flow", ("pause",))

        # 18-buffer system: 3 groups × 6 buffers each
        # Using 6 batches per group for better VALU utilization

        def alloc_buffer_set(name):
            """Allocate a complete buffer set for one batch."""
            return {
                'v_idx': self.alloc_scratch(f"v_idx_{name}", VLEN),
                'v_val': self.alloc_scratch(f"v_val_{name}", VLEN),
                'v_node_val': self.alloc_scratch(f"v_node_val_{name}", VLEN),
                'v_tmp1': self.alloc_scratch(f"v_tmp1_{name}", VLEN),
                'v_tmp2': self.alloc_scratch(f"v_tmp2_{name}", VLEN),
                'node_addrs': [self.alloc_scratch(f"node_addr_{name}_{j}") for j in range(VLEN)],
                'addr_idx': self.alloc_scratch(f"addr_idx_{name}"),
                'addr_val': self.alloc_scratch(f"addr_val_{name}"),
            }

        # Allocate 18 buffer sets (3 groups of 6)
        buf = [alloc_buffer_set(chr(ord('A') + i) if i < 26 else f"B{i-26}") for i in range(18)]

        # Vector constants
        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)

        # Pre-broadcast all hash constants with optimized batching
        # First, collect all constants we need and allocate vectors
        hash_v_consts = []
        const_broadcasts = []  # List of (scalar_addr, vector_addr) pairs

        for stage_idx, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if op1 == "+" and op2 == "+" and op3 == "<<":
                # multiply_add optimization
                multiplier = 1 + (1 << val3)
                v_mult = self.alloc_scratch(f"v_hash_mult_{stage_idx}", VLEN)
                v_c1 = self.alloc_scratch(f"v_hash_c1_{val1}", VLEN)
                const_broadcasts.append((self.scratch_const(multiplier), v_mult))
                const_broadcasts.append((self.scratch_const(val1), v_c1))
                hash_v_consts.append(("multiply_add", v_mult, v_c1, None, None))
            else:
                v_c1 = self.alloc_scratch(f"v_hash_c1_{val1}", VLEN)
                v_c2 = self.alloc_scratch(f"v_hash_c2_{val3}", VLEN)
                const_broadcasts.append((self.scratch_const(val1), v_c1))
                const_broadcasts.append((self.scratch_const(val3), v_c2))
                hash_v_consts.append((op1, v_c1, op2, op3, v_c2))

        # Add general constants
        const_broadcasts.append((zero_const, v_zero))
        const_broadcasts.append((one_const, v_one))
        const_broadcasts.append((two_const, v_two))
        const_broadcasts.append((n_nodes_const, v_n_nodes))

        # Store broadcasts for later merging with prologue load
        # This saves cycles by doing VALU broadcasts during Load cycles
        broadcast_instrs = []
        for i in range(0, len(const_broadcasts), 6):
            batch = const_broadcasts[i:i+6]
            broadcast_instrs.append({"valu": [("vbroadcast", vec, scalar) for scalar, vec in batch]})

        # Build batch offset list
        batches_per_round = batch_size // VLEN
        total_batches = batches_per_round * rounds
        batch_offsets = [(b % batches_per_round) * VLEN for b in range(total_batches)]

        def emit_load_full(offset, b):
            """Emit full load for a buffer."""
            offset_const = self.scratch_const(offset)
            instrs = [
                {"alu": [
                    ("+", b['addr_idx'], self.scratch["inp_indices_p"], offset_const),
                    ("+", b['addr_val'], self.scratch["inp_values_p"], offset_const),
                ]},
                {"load": [("vload", b['v_idx'], b['addr_idx']), ("vload", b['v_val'], b['addr_val'])]},
                {"alu": [("+", b['node_addrs'][j], self.scratch["forest_values_p"], b['v_idx'] + j) for j in range(VLEN)]},
            ]
            for j in range(0, VLEN, 2):
                instrs.append({"load": [
                    ("load", b['v_node_val'] + j, b['node_addrs'][j]),
                    ("load", b['v_node_val'] + j + 1, b['node_addrs'][j + 1]),
                ]})
            return instrs

        def emit_load_3_parallel(offsets, bufs):
            """Emit loads for 3 batches with optimized interleaving.
            
            Key insight: After computing batch N's node addresses, we can start
            loading batch N's node values while computing batch N+1's addresses.
            This overlaps ALU with Load operations.
            """
            b0, b1, b2 = bufs
            off0, off1, off2 = [self.scratch_const(o) for o in offsets]
            
            instrs = []
            
            # Phase 1: Compute all memory addresses (ALU) - 6 ops, 1 cycle
            instrs.append({"alu": [
                ("+", b0['addr_idx'], self.scratch["inp_indices_p"], off0),
                ("+", b0['addr_val'], self.scratch["inp_values_p"], off0),
                ("+", b1['addr_idx'], self.scratch["inp_indices_p"], off1),
                ("+", b1['addr_val'], self.scratch["inp_values_p"], off1),
                ("+", b2['addr_idx'], self.scratch["inp_indices_p"], off2),
                ("+", b2['addr_val'], self.scratch["inp_values_p"], off2),
            ]})
            
            # Phase 2: Vector loads for idx/val (6 loads = 3 cycles at 2 loads/cycle)
            instrs.append({"load": [("vload", b0['v_idx'], b0['addr_idx']), ("vload", b0['v_val'], b0['addr_val'])]})
            instrs.append({"load": [("vload", b1['v_idx'], b1['addr_idx']), ("vload", b1['v_val'], b1['addr_val'])]})
            instrs.append({"load": [("vload", b2['v_idx'], b2['addr_idx']), ("vload", b2['v_val'], b2['addr_val'])]})
            
            # Phase 3: Interleave node_addr computation with node value loads
            # Batch 0 node addrs (8 ALU ops)
            instrs.append({"alu": [("+", b0['node_addrs'][j], self.scratch["forest_values_p"], b0['v_idx'] + j) for j in range(VLEN)]})
            
            # Batch 0 first 2 loads + batch 1 node addrs (8 ALU ops)
            instrs.append({"load": [
                ("load", b0['v_node_val'] + 0, b0['node_addrs'][0]),
                ("load", b0['v_node_val'] + 1, b0['node_addrs'][1]),
            ], "alu": [("+", b1['node_addrs'][j], self.scratch["forest_values_p"], b1['v_idx'] + j) for j in range(VLEN)]})
            
            # Batch 0 next 2 loads + batch 2 node addrs (8 ALU ops)
            instrs.append({"load": [
                ("load", b0['v_node_val'] + 2, b0['node_addrs'][2]),
                ("load", b0['v_node_val'] + 3, b0['node_addrs'][3]),
            ], "alu": [("+", b2['node_addrs'][j], self.scratch["forest_values_p"], b2['v_idx'] + j) for j in range(VLEN)]})
            
            # Batch 0 last 2 loads + batch 1 first 2 loads
            instrs.append({"load": [
                ("load", b0['v_node_val'] + 4, b0['node_addrs'][4]),
                ("load", b0['v_node_val'] + 5, b0['node_addrs'][5]),
            ]})
            instrs.append({"load": [
                ("load", b0['v_node_val'] + 6, b0['node_addrs'][6]),
                ("load", b0['v_node_val'] + 7, b0['node_addrs'][7]),
            ]})
            
            # Remaining batch 1 and batch 2 scalar loads
            for j in range(0, VLEN, 2):
                instrs.append({"load": [
                    ("load", b1['v_node_val'] + j, b1['node_addrs'][j]),
                    ("load", b1['v_node_val'] + j + 1, b1['node_addrs'][j + 1]),
                ]})
            for j in range(0, VLEN, 2):
                instrs.append({"load": [
                    ("load", b2['v_node_val'] + j, b2['node_addrs'][j]),
                    ("load", b2['v_node_val'] + j + 1, b2['node_addrs'][j + 1]),
                ]})
            
            return instrs

        def emit_store(offset, b):
            """Emit store for a buffer."""
            offset_const = self.scratch_const(offset)
            return [
                {"alu": [
                    ("+", b['addr_idx'], self.scratch["inp_indices_p"], offset_const),
                    ("+", b['addr_val'], self.scratch["inp_values_p"], offset_const),
                ]},
                {"store": [("vstore", b['addr_idx'], b['v_idx']), ("vstore", b['addr_val'], b['v_val'])]},
            ]

        def emit_load_4_parallel(offsets, bufs):
            """Emit loads for 4 batches with interleaving."""
            instrs = []
            off_consts = [self.scratch_const(o) for o in offsets]

            # Compute all addresses (8 ALU ops)
            instrs.append({"alu": [
                ("+", bufs[i]['addr_idx'], self.scratch["inp_indices_p"], off_consts[i])
                for i in range(4)
            ] + [
                ("+", bufs[i]['addr_val'], self.scratch["inp_values_p"], off_consts[i])
                for i in range(4)
            ]})

            # Vector loads (4 batches × 2 vloads = 4 cycles)
            instrs.append({"load": [("vload", bufs[0]['v_idx'], bufs[0]['addr_idx']),
                                    ("vload", bufs[0]['v_val'], bufs[0]['addr_val'])]})
            for i in range(1, 4):
                instrs.append({"load": [("vload", bufs[i]['v_idx'], bufs[i]['addr_idx']),
                                        ("vload", bufs[i]['v_val'], bufs[i]['addr_val'])],
                               "alu": [("+", bufs[i-1]['node_addrs'][j], self.scratch["forest_values_p"], bufs[i-1]['v_idx'] + j) for j in range(VLEN)]})
            instrs.append({"alu": [("+", bufs[3]['node_addrs'][j], self.scratch["forest_values_p"], bufs[3]['v_idx'] + j) for j in range(VLEN)]})

            # Scalar loads (4 batches × 4 cycles = 16 cycles)
            for batch_idx in range(4):
                for j in range(0, VLEN, 2):
                    instrs.append({"load": [
                        ("load", bufs[batch_idx]['v_node_val'] + j, bufs[batch_idx]['node_addrs'][j]),
                        ("load", bufs[batch_idx]['v_node_val'] + j + 1, bufs[batch_idx]['node_addrs'][j + 1]),
                    ]})
            return instrs

        def emit_store_4(offsets, bufs):
            """Emit stores for 4 batches."""
            instrs = []
            off_consts = [self.scratch_const(o) for o in offsets]
            instrs.append({"alu": [
                ("+", bufs[i]['addr_idx'], self.scratch["inp_indices_p"], off_consts[i])
                for i in range(4)
            ] + [
                ("+", bufs[i]['addr_val'], self.scratch["inp_values_p"], off_consts[i])
                for i in range(4)
            ]})
            for i in range(4):
                instrs.append({"store": [("vstore", bufs[i]['addr_idx'], bufs[i]['v_idx']),
                                         ("vstore", bufs[i]['addr_val'], bufs[i]['v_val'])]})
            return instrs

        def emit_load_6_parallel(offsets, bufs):
            """Emit loads for 6 batches with deeply interleaved operations.

            Key optimization: Start scalar loads as soon as node_addrs are computed,
            overlapping with remaining vloads and addr computations.
            """
            instrs = []
            off_consts = [self.scratch_const(o) for o in offsets]

            # Phase 1: Compute all base addresses (12 ALU ops, 1 cycle)
            instrs.append({"alu": [
                ("+", bufs[i]['addr_idx'], self.scratch["inp_indices_p"], off_consts[i])
                for i in range(6)
            ] + [
                ("+", bufs[i]['addr_val'], self.scratch["inp_values_p"], off_consts[i])
                for i in range(6)
            ]})

            # Phase 2: Deeply interleaved loading
            # Cycle 1: vload B0
            instrs.append({"load": [("vload", bufs[0]['v_idx'], bufs[0]['addr_idx']),
                                    ("vload", bufs[0]['v_val'], bufs[0]['addr_val'])]})

            # Cycle 2: vload B1, compute B0's node_addrs
            instrs.append({"load": [("vload", bufs[1]['v_idx'], bufs[1]['addr_idx']),
                                    ("vload", bufs[1]['v_val'], bufs[1]['addr_val'])],
                           "alu": [("+", bufs[0]['node_addrs'][j], self.scratch["forest_values_p"], bufs[0]['v_idx'] + j) for j in range(VLEN)]})

            # Cycle 3: vload B2, compute B1's node_addrs - B0's scalar loads can now start!
            # But we can only do 2 loads per cycle, so we need to continue in following cycles
            instrs.append({"load": [("vload", bufs[2]['v_idx'], bufs[2]['addr_idx']),
                                    ("vload", bufs[2]['v_val'], bufs[2]['addr_val'])],
                           "alu": [("+", bufs[1]['node_addrs'][j], self.scratch["forest_values_p"], bufs[1]['v_idx'] + j) for j in range(VLEN)]})

            # Cycle 4: vload B3, compute B2's node_addrs
            instrs.append({"load": [("vload", bufs[3]['v_idx'], bufs[3]['addr_idx']),
                                    ("vload", bufs[3]['v_val'], bufs[3]['addr_val'])],
                           "alu": [("+", bufs[2]['node_addrs'][j], self.scratch["forest_values_p"], bufs[2]['v_idx'] + j) for j in range(VLEN)]})

            # Cycle 5: vload B4, compute B3's node_addrs
            instrs.append({"load": [("vload", bufs[4]['v_idx'], bufs[4]['addr_idx']),
                                    ("vload", bufs[4]['v_val'], bufs[4]['addr_val'])],
                           "alu": [("+", bufs[3]['node_addrs'][j], self.scratch["forest_values_p"], bufs[3]['v_idx'] + j) for j in range(VLEN)]})

            # Cycle 6: vload B5, compute B4's node_addrs
            instrs.append({"load": [("vload", bufs[5]['v_idx'], bufs[5]['addr_idx']),
                                    ("vload", bufs[5]['v_val'], bufs[5]['addr_val'])],
                           "alu": [("+", bufs[4]['node_addrs'][j], self.scratch["forest_values_p"], bufs[4]['v_idx'] + j) for j in range(VLEN)]})

            # Cycle 7: compute B5's node_addrs, start B0's scalar loads
            instrs.append({"load": [
                ("load", bufs[0]['v_node_val'] + 0, bufs[0]['node_addrs'][0]),
                ("load", bufs[0]['v_node_val'] + 1, bufs[0]['node_addrs'][1]),
            ], "alu": [("+", bufs[5]['node_addrs'][j], self.scratch["forest_values_p"], bufs[5]['v_idx'] + j) for j in range(VLEN)]})

            # Remaining scalar loads: 46 loads at 2/cycle = 23 cycles
            # B0: 6 more loads (indices 2-7)
            # B1-B5: 8 loads each = 40 loads
            remaining_loads = []
            for j in range(2, VLEN, 2):
                remaining_loads.append((0, j))
            for batch_idx in range(1, 6):
                for j in range(0, VLEN, 2):
                    remaining_loads.append((batch_idx, j))

            for batch_idx, j in remaining_loads:
                instrs.append({"load": [
                    ("load", bufs[batch_idx]['v_node_val'] + j, bufs[batch_idx]['node_addrs'][j]),
                    ("load", bufs[batch_idx]['v_node_val'] + j + 1, bufs[batch_idx]['node_addrs'][j + 1]),
                ]})

            return instrs

        def emit_store_6(offsets, bufs):
            """Emit stores for 6 batches."""
            instrs = []
            # Compute all addresses first
            off_consts = [self.scratch_const(o) for o in offsets]
            instrs.append({"alu": [
                ("+", bufs[i]['addr_idx'], self.scratch["inp_indices_p"], off_consts[i])
                for i in range(6)
            ] + [
                ("+", bufs[i]['addr_val'], self.scratch["inp_values_p"], off_consts[i])
                for i in range(6)
            ]})
            # Do stores (2 per cycle)
            for i in range(0, 6, 1):
                instrs.append({"store": [("vstore", bufs[i]['addr_idx'], bufs[i]['v_idx']),
                                         ("vstore", bufs[i]['addr_val'], bufs[i]['v_val'])]})
            return instrs

        def emit_compute_3_parallel(b0, b1, b2):
            """Emit compute for 3 batches in PARALLEL using all 6 VALU slots!"""
            instrs = []

            # XOR for all 3 batches
            instrs.append({"valu": [
                ("^", b0['v_val'], b0['v_val'], b0['v_node_val']),
                ("^", b1['v_val'], b1['v_val'], b1['v_node_val']),
                ("^", b2['v_val'], b2['v_val'], b2['v_node_val']),
            ]})

            # Hash stages - use multiply_add optimization where possible!
            for stage_info in hash_v_consts:
                if stage_info[0] == "multiply_add":
                    # Optimized: val = val * multiplier + constant (1 VALU cycle instead of 2!)
                    _, v_mult, v_c1, _, _ = stage_info
                    instrs.append({"valu": [
                        ("multiply_add", b0['v_val'], b0['v_val'], v_mult, v_c1),
                        ("multiply_add", b1['v_val'], b1['v_val'], v_mult, v_c1),
                        ("multiply_add", b2['v_val'], b2['v_val'], v_mult, v_c1),
                    ]})
                else:
                    # Standard: 2 cycles
                    op1, v_c1, op2, op3, v_c2 = stage_info
                    instrs.append({"valu": [
                        (op1, b0['v_tmp1'], b0['v_val'], v_c1), (op3, b0['v_tmp2'], b0['v_val'], v_c2),
                        (op1, b1['v_tmp1'], b1['v_val'], v_c1), (op3, b1['v_tmp2'], b1['v_val'], v_c2),
                        (op1, b2['v_tmp1'], b2['v_val'], v_c1), (op3, b2['v_tmp2'], b2['v_val'], v_c2),
                    ]})
                    instrs.append({"valu": [
                        (op2, b0['v_val'], b0['v_tmp1'], b0['v_tmp2']),
                        (op2, b1['v_val'], b1['v_tmp1'], b1['v_tmp2']),
                        (op2, b2['v_val'], b2['v_tmp1'], b2['v_tmp2']),
                    ]})

            # idx computation using multiply_add: idx = idx*2 + (val&1) + 1
            # Step 1: tmp = val & 1
            # Step 2: tmp = tmp + 1
            # Step 3: idx = multiply_add(idx, 2, tmp) = idx*2 + tmp
            instrs.append({"valu": [
                ("&", b0['v_tmp1'], b0['v_val'], v_one),
                ("&", b1['v_tmp1'], b1['v_val'], v_one),
                ("&", b2['v_tmp1'], b2['v_val'], v_one),
            ]})
            instrs.append({"valu": [
                ("+", b0['v_tmp1'], b0['v_tmp1'], v_one),
                ("+", b1['v_tmp1'], b1['v_tmp1'], v_one),
                ("+", b2['v_tmp1'], b2['v_tmp1'], v_one),
            ]})
            instrs.append({"valu": [
                ("multiply_add", b0['v_idx'], b0['v_idx'], v_two, b0['v_tmp1']),
                ("multiply_add", b1['v_idx'], b1['v_idx'], v_two, b1['v_tmp1']),
                ("multiply_add", b2['v_idx'], b2['v_idx'], v_two, b2['v_tmp1']),
            ]})

            # Bounds check
            instrs.append({"valu": [
                ("<", b0['v_tmp1'], b0['v_idx'], v_n_nodes),
                ("<", b1['v_tmp1'], b1['v_idx'], v_n_nodes),
                ("<", b2['v_tmp1'], b2['v_idx'], v_n_nodes),
            ]})

            # Use VALU multiply instead of flow vselect (idx * cond)
            instrs.append({"valu": [
                ("*", b0['v_idx'], b0['v_idx'], b0['v_tmp1']),
                ("*", b1['v_idx'], b1['v_idx'], b1['v_tmp1']),
                ("*", b2['v_idx'], b2['v_idx'], b2['v_tmp1']),
            ]})

            return instrs

        def emit_compute_4_parallel(batches):
            """Emit compute for 4 batches using 4-6 VALU slots."""
            b0, b1, b2, b3 = batches
            instrs = []

            # XOR for all 4 batches
            instrs.append({"valu": [
                ("^", b0['v_val'], b0['v_val'], b0['v_node_val']),
                ("^", b1['v_val'], b1['v_val'], b1['v_node_val']),
                ("^", b2['v_val'], b2['v_val'], b2['v_node_val']),
                ("^", b3['v_val'], b3['v_val'], b3['v_node_val']),
            ]})

            # Hash stages
            for stage_info in hash_v_consts:
                if stage_info[0] == "multiply_add":
                    _, v_mult, v_c1, _, _ = stage_info
                    instrs.append({"valu": [
                        ("multiply_add", b0['v_val'], b0['v_val'], v_mult, v_c1),
                        ("multiply_add", b1['v_val'], b1['v_val'], v_mult, v_c1),
                        ("multiply_add", b2['v_val'], b2['v_val'], v_mult, v_c1),
                        ("multiply_add", b3['v_val'], b3['v_val'], v_mult, v_c1),
                    ]})
                else:
                    # 2 cycles: op1+op3 for all 4 (8 ops, 2 cycles), then op2 for all 4
                    op1, v_c1, op2, op3, v_c2 = stage_info
                    instrs.append({"valu": [
                        (op1, b0['v_tmp1'], b0['v_val'], v_c1), (op3, b0['v_tmp2'], b0['v_val'], v_c2),
                        (op1, b1['v_tmp1'], b1['v_val'], v_c1), (op3, b1['v_tmp2'], b1['v_val'], v_c2),
                        (op1, b2['v_tmp1'], b2['v_val'], v_c1), (op3, b2['v_tmp2'], b2['v_val'], v_c2),
                    ]})
                    instrs.append({"valu": [
                        (op1, b3['v_tmp1'], b3['v_val'], v_c1), (op3, b3['v_tmp2'], b3['v_val'], v_c2),
                        (op2, b0['v_val'], b0['v_tmp1'], b0['v_tmp2']),
                        (op2, b1['v_val'], b1['v_tmp1'], b1['v_tmp2']),
                        (op2, b2['v_val'], b2['v_tmp1'], b2['v_tmp2']),
                    ]})
                    instrs.append({"valu": [
                        (op2, b3['v_val'], b3['v_tmp1'], b3['v_tmp2']),
                    ]})

            # idx computation
            instrs.append({"valu": [
                ("&", b0['v_tmp1'], b0['v_val'], v_one),
                ("&", b1['v_tmp1'], b1['v_val'], v_one),
                ("&", b2['v_tmp1'], b2['v_val'], v_one),
                ("&", b3['v_tmp1'], b3['v_val'], v_one),
            ]})
            instrs.append({"valu": [
                ("+", b0['v_tmp1'], b0['v_tmp1'], v_one),
                ("+", b1['v_tmp1'], b1['v_tmp1'], v_one),
                ("+", b2['v_tmp1'], b2['v_tmp1'], v_one),
                ("+", b3['v_tmp1'], b3['v_tmp1'], v_one),
            ]})
            instrs.append({"valu": [
                ("multiply_add", b0['v_idx'], b0['v_idx'], v_two, b0['v_tmp1']),
                ("multiply_add", b1['v_idx'], b1['v_idx'], v_two, b1['v_tmp1']),
                ("multiply_add", b2['v_idx'], b2['v_idx'], v_two, b2['v_tmp1']),
                ("multiply_add", b3['v_idx'], b3['v_idx'], v_two, b3['v_tmp1']),
            ]})

            # Bounds check and multiply
            instrs.append({"valu": [
                ("<", b0['v_tmp1'], b0['v_idx'], v_n_nodes),
                ("<", b1['v_tmp1'], b1['v_idx'], v_n_nodes),
                ("<", b2['v_tmp1'], b2['v_idx'], v_n_nodes),
                ("<", b3['v_tmp1'], b3['v_idx'], v_n_nodes),
            ]})
            instrs.append({"valu": [
                ("*", b0['v_idx'], b0['v_idx'], b0['v_tmp1']),
                ("*", b1['v_idx'], b1['v_idx'], b1['v_tmp1']),
                ("*", b2['v_idx'], b2['v_idx'], b2['v_tmp1']),
                ("*", b3['v_idx'], b3['v_idx'], b3['v_tmp1']),
            ]})

            return instrs

        def emit_compute_6_parallel(batches):
            """Emit compute for 6 batches using all 6 VALU slots with split processing."""
            b0, b1, b2, b3, b4, b5 = batches
            instrs = []

            # XOR for all 6 batches (1 cycle)
            instrs.append({"valu": [
                ("^", b0['v_val'], b0['v_val'], b0['v_node_val']),
                ("^", b1['v_val'], b1['v_val'], b1['v_node_val']),
                ("^", b2['v_val'], b2['v_val'], b2['v_node_val']),
                ("^", b3['v_val'], b3['v_val'], b3['v_node_val']),
                ("^", b4['v_val'], b4['v_val'], b4['v_node_val']),
                ("^", b5['v_val'], b5['v_val'], b5['v_node_val']),
            ]})

            # Hash stages
            for stage_info in hash_v_consts:
                if stage_info[0] == "multiply_add":
                    # Optimized: 1 cycle for 6 batches
                    _, v_mult, v_c1, _, _ = stage_info
                    instrs.append({"valu": [
                        ("multiply_add", b0['v_val'], b0['v_val'], v_mult, v_c1),
                        ("multiply_add", b1['v_val'], b1['v_val'], v_mult, v_c1),
                        ("multiply_add", b2['v_val'], b2['v_val'], v_mult, v_c1),
                        ("multiply_add", b3['v_val'], b3['v_val'], v_mult, v_c1),
                        ("multiply_add", b4['v_val'], b4['v_val'], v_mult, v_c1),
                        ("multiply_add", b5['v_val'], b5['v_val'], v_mult, v_c1),
                    ]})
                else:
                    # Non-optimized: 3 cycles for 6 batches
                    # Cycle 1: op1+op3 for batches 0-2 (6 ops)
                    # Cycle 2: op1+op3 for batches 3-5 (6 ops)
                    # Cycle 3: op2 for all 6 batches (6 ops)
                    op1, v_c1, op2, op3, v_c2 = stage_info
                    instrs.append({"valu": [
                        (op1, b0['v_tmp1'], b0['v_val'], v_c1), (op3, b0['v_tmp2'], b0['v_val'], v_c2),
                        (op1, b1['v_tmp1'], b1['v_val'], v_c1), (op3, b1['v_tmp2'], b1['v_val'], v_c2),
                        (op1, b2['v_tmp1'], b2['v_val'], v_c1), (op3, b2['v_tmp2'], b2['v_val'], v_c2),
                    ]})
                    instrs.append({"valu": [
                        (op1, b3['v_tmp1'], b3['v_val'], v_c1), (op3, b3['v_tmp2'], b3['v_val'], v_c2),
                        (op1, b4['v_tmp1'], b4['v_val'], v_c1), (op3, b4['v_tmp2'], b4['v_val'], v_c2),
                        (op1, b5['v_tmp1'], b5['v_val'], v_c1), (op3, b5['v_tmp2'], b5['v_val'], v_c2),
                    ]})
                    instrs.append({"valu": [
                        (op2, b0['v_val'], b0['v_tmp1'], b0['v_tmp2']),
                        (op2, b1['v_val'], b1['v_tmp1'], b1['v_tmp2']),
                        (op2, b2['v_val'], b2['v_tmp1'], b2['v_tmp2']),
                        (op2, b3['v_val'], b3['v_tmp1'], b3['v_tmp2']),
                        (op2, b4['v_val'], b4['v_tmp1'], b4['v_tmp2']),
                        (op2, b5['v_val'], b5['v_tmp1'], b5['v_tmp2']),
                    ]})

            # idx computation for all 6 batches (3 cycles)
            instrs.append({"valu": [
                ("&", b0['v_tmp1'], b0['v_val'], v_one),
                ("&", b1['v_tmp1'], b1['v_val'], v_one),
                ("&", b2['v_tmp1'], b2['v_val'], v_one),
                ("&", b3['v_tmp1'], b3['v_val'], v_one),
                ("&", b4['v_tmp1'], b4['v_val'], v_one),
                ("&", b5['v_tmp1'], b5['v_val'], v_one),
            ]})
            instrs.append({"valu": [
                ("+", b0['v_tmp1'], b0['v_tmp1'], v_one),
                ("+", b1['v_tmp1'], b1['v_tmp1'], v_one),
                ("+", b2['v_tmp1'], b2['v_tmp1'], v_one),
                ("+", b3['v_tmp1'], b3['v_tmp1'], v_one),
                ("+", b4['v_tmp1'], b4['v_tmp1'], v_one),
                ("+", b5['v_tmp1'], b5['v_tmp1'], v_one),
            ]})
            instrs.append({"valu": [
                ("multiply_add", b0['v_idx'], b0['v_idx'], v_two, b0['v_tmp1']),
                ("multiply_add", b1['v_idx'], b1['v_idx'], v_two, b1['v_tmp1']),
                ("multiply_add", b2['v_idx'], b2['v_idx'], v_two, b2['v_tmp1']),
                ("multiply_add", b3['v_idx'], b3['v_idx'], v_two, b3['v_tmp1']),
                ("multiply_add", b4['v_idx'], b4['v_idx'], v_two, b4['v_tmp1']),
                ("multiply_add", b5['v_idx'], b5['v_idx'], v_two, b5['v_tmp1']),
            ]})

            # Bounds check (1 cycle)
            instrs.append({"valu": [
                ("<", b0['v_tmp1'], b0['v_idx'], v_n_nodes),
                ("<", b1['v_tmp1'], b1['v_idx'], v_n_nodes),
                ("<", b2['v_tmp1'], b2['v_idx'], v_n_nodes),
                ("<", b3['v_tmp1'], b3['v_idx'], v_n_nodes),
                ("<", b4['v_tmp1'], b4['v_idx'], v_n_nodes),
                ("<", b5['v_tmp1'], b5['v_idx'], v_n_nodes),
            ]})

            # Multiply (1 cycle)
            instrs.append({"valu": [
                ("*", b0['v_idx'], b0['v_idx'], b0['v_tmp1']),
                ("*", b1['v_idx'], b1['v_idx'], b1['v_tmp1']),
                ("*", b2['v_idx'], b2['v_idx'], b2['v_tmp1']),
                ("*", b3['v_idx'], b3['v_idx'], b3['v_tmp1']),
                ("*", b4['v_idx'], b4['v_idx'], b4['v_tmp1']),
                ("*", b5['v_idx'], b5['v_idx'], b5['v_tmp1']),
            ]})

            return instrs

        def emit_compute_single(b):
            """Emit compute for a single batch."""
            instrs = [{"valu": [("^", b['v_val'], b['v_val'], b['v_node_val'])]}]

            for stage_info in hash_v_consts:
                if stage_info[0] == "multiply_add":
                    _, v_mult, v_c1, _, _ = stage_info
                    instrs.append({"valu": [("multiply_add", b['v_val'], b['v_val'], v_mult, v_c1)]})
                else:
                    op1, v_c1, op2, op3, v_c2 = stage_info
                    instrs.append({"valu": [(op1, b['v_tmp1'], b['v_val'], v_c1), (op3, b['v_tmp2'], b['v_val'], v_c2)]})
                    instrs.append({"valu": [(op2, b['v_val'], b['v_tmp1'], b['v_tmp2'])]})

            instrs.extend([
                {"valu": [("&", b['v_tmp1'], b['v_val'], v_one)]},
                {"valu": [("+", b['v_tmp1'], b['v_tmp1'], v_one)]},
                {"valu": [("multiply_add", b['v_idx'], b['v_idx'], v_two, b['v_tmp1'])]},
                {"valu": [("<", b['v_tmp1'], b['v_idx'], v_n_nodes)]},
                {"valu": [("*", b['v_idx'], b['v_idx'], b['v_tmp1'])]},
            ])
            return instrs

        def interleave_phases(compute_instrs, load_instrs, store_instrs):
            """
            Interleave compute (VALU) with load (ALU/Load) and store (ALU/Store).
            IMPORTANT: Instructions are processed in order to maintain dependencies.
            Alternates between trying store and load first to spread them evenly.
            This ensures the final compute cycle can still merge with a load instruction.
            """
            result = []
            ci, li, si = 0, 0, 0
            store_first = True  # Alternate which to try first
            
            # Slot limits
            LIMITS = {"alu": 12, "valu": 6, "load": 2, "store": 2, "flow": 1}

            def can_add(merged, instr):
                """Check if instr can be added to merged without exceeding limits."""
                for eng, slots in instr.items():
                    current = len(merged.get(eng, []))
                    if current + len(slots) > LIMITS.get(eng, 0):
                        return False
                return True

            def add_instr(merged, instr):
                """Add instr to merged."""
                for eng, slots in instr.items():
                    if eng not in merged:
                        merged[eng] = []
                    merged[eng].extend(slots)

            while ci < len(compute_instrs) or li < len(load_instrs) or si < len(store_instrs):
                merged = {}

                # Add compute first (always succeeds for one VALU instruction)
                if ci < len(compute_instrs):
                    add_instr(merged, compute_instrs[ci])
                    ci += 1

                # Alternate priority between store and load to spread them evenly
                if store_first:
                    # Try store first
                    if si < len(store_instrs) and can_add(merged, store_instrs[si]):
                        add_instr(merged, store_instrs[si])
                        si += 1
                    # Then try load
                    if li < len(load_instrs) and can_add(merged, load_instrs[li]):
                        add_instr(merged, load_instrs[li])
                        li += 1
                else:
                    # Try load first
                    if li < len(load_instrs) and can_add(merged, load_instrs[li]):
                        add_instr(merged, load_instrs[li])
                        li += 1
                    # Then try store
                    if si < len(store_instrs) and can_add(merged, store_instrs[si]):
                        add_instr(merged, store_instrs[si])
                        si += 1
                
                store_first = not store_first  # Alternate for next cycle

                if merged:
                    result.append(merged)

            # Remaining instructions - MERGE stores with loads!
            while si < len(store_instrs) or li < len(load_instrs):
                merged = {}
                # Try to add a store
                if si < len(store_instrs) and can_add(merged, store_instrs[si]):
                    add_instr(merged, store_instrs[si])
                    si += 1
                # Try to add a load
                if li < len(load_instrs) and can_add(merged, load_instrs[li]):
                    add_instr(merged, load_instrs[li])
                    li += 1
                if merged:
                    result.append(merged)

            return result

        body_instrs = []

        if total_batches == 0:
            pass
        elif total_batches < 6:
            # Less than 6 batches - use single batch processing
            for batch_idx in range(total_batches):
                b = buf[batch_idx]
                body_instrs.extend(emit_load_full(batch_offsets[batch_idx], b))
                body_instrs.extend(emit_compute_single(b))
                body_instrs.extend(emit_store(batch_offsets[batch_idx], b))
        else:
            # 3-phase pipelining with 18 buffers (3 groups of 6 batches)
            BATCHES_PER_GROUP = 6
            num_groups = total_batches // BATCHES_PER_GROUP
            remainder = total_batches % BATCHES_PER_GROUP

            def get_group_bufs(group_idx):
                base = (group_idx % 3) * BATCHES_PER_GROUP
                return buf[base:base+BATCHES_PER_GROUP]

            # Prologue: Load first group, merged with constant broadcasts
            prologue_offsets = [batch_offsets[i] for i in range(BATCHES_PER_GROUP)]
            prologue_load_instrs = emit_load_6_parallel(prologue_offsets, buf[0:BATCHES_PER_GROUP])

            # Merge broadcasts with prologue loads to save cycles
            bi = 0
            for li, load_instr in enumerate(prologue_load_instrs):
                if bi < len(broadcast_instrs):
                    # Merge broadcast VALU with load instruction
                    for eng, slots in broadcast_instrs[bi].items():
                        if eng not in load_instr:
                            load_instr[eng] = []
                        load_instr[eng].extend(slots)
                    bi += 1
                body_instrs.append(load_instr)

            # Append any remaining broadcasts
            while bi < len(broadcast_instrs):
                body_instrs.append(broadcast_instrs[bi])
                bi += 1

            # Main loop: Compute(G) + Load(G+1), then Store(G) merged with remaining Load(G+1)
            # This structure ensures stores happen AFTER compute and merge with load tail
            for group in range(num_groups):
                compute_bufs = get_group_bufs(group)
                compute_instrs = emit_compute_6_parallel(compute_bufs)

                # Load next group
                load_instrs = []
                if group < num_groups - 1:
                    load_bufs = get_group_bufs(group + 1)
                    load_base = (group + 1) * BATCHES_PER_GROUP
                    load_offsets = [batch_offsets[load_base + i] for i in range(BATCHES_PER_GROUP)]
                    load_instrs = emit_load_6_parallel(load_offsets, load_bufs)

                # Store current group AFTER compute (can merge with remaining loads)
                curr_store_instrs = emit_store_6(
                    [batch_offsets[group * BATCHES_PER_GROUP + i] for i in range(BATCHES_PER_GROUP)],
                    compute_bufs
                )

                # Custom interleave: compute first, then store+remaining_loads
                result = []
                ci, li = 0, 0

                # Phase 1: Compute + Load (as much as fits)
                while ci < len(compute_instrs):
                    merged = dict(compute_instrs[ci])
                    ci += 1
                    # Try to add loads
                    if li < len(load_instrs):
                        for eng, slots in load_instrs[li].items():
                            if eng not in merged:
                                merged[eng] = []
                            merged[eng].extend(slots)
                        li += 1
                    result.append(merged)

                # Phase 2: Store + remaining Load (after compute)
                si = 0
                while si < len(curr_store_instrs) or li < len(load_instrs):
                    merged = {}
                    if si < len(curr_store_instrs):
                        for eng, slots in curr_store_instrs[si].items():
                            if eng not in merged:
                                merged[eng] = []
                            merged[eng].extend(slots)
                        si += 1
                    if li < len(load_instrs):
                        for eng, slots in load_instrs[li].items():
                            if eng not in merged:
                                merged[eng] = []
                            merged[eng].extend(slots)
                        li += 1
                    if merged:
                        result.append(merged)

                body_instrs.extend(result)

            # Handle remainder (2 batches for 512/6)
            remainder_start = num_groups * BATCHES_PER_GROUP
            for i in range(remainder):
                batch_idx = remainder_start + i
                b = buf[i]
                body_instrs.extend(emit_load_full(batch_offsets[batch_idx], b))
                body_instrs.extend(emit_compute_single(b))
                body_instrs.extend(emit_store(batch_offsets[batch_idx], b))

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
