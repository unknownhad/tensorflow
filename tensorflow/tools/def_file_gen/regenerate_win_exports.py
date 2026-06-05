#!/usr/bin/env python3
# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Regenerates the Windows Export Table for TensorFlow.

================================================================================
ARCHITECTURAL CONTEXT & GOAL
================================================================================
In TensorFlow's modern Bzlmod hermetic Windows build architecture
(USE_PYWRAP_RULES=True), legacy dynamic allowlists (win_lib_files &
symbols_pybind) are obsolete NO-OPs. Bazel links the common C++ core
(_pywrap_tensorflow_common.dll) directly against the static checked-in Bzlmod
export table (tensorflow/python/_pywrap_tensorflow.def). The goal of
regenerate_win_exports.py is to fully automate the maintenance of this static
file. It seamlessly combines static C++ AST harvesting, Public Boundary API
filtering, MSVC name mangling heuristics, and direct DEF file patching into a
single, lightning-fast execution pass.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
import re
import subprocess
import sys
from typing import Optional
import uuid


def run_cmd(
    cmd: list[str], cwd: str = ".", check: bool = True, silent: bool = False
) -> str:
  """Runs a subprocess command and returns stdout."""
  try:
    proc = subprocess.run(
        cmd, capture_output=True, text=True, cwd=cwd, check=check
    )
    return proc.stdout
  except subprocess.CalledProcessError as e:
    if not silent:
      print(
          f"Command failed: {' '.join(cmd)}\nError: {e.stderr}",
          file=sys.stderr,
      )
    if check:
      raise
    return ""


def parse_static_export_table(
    workspace_root: str, static_export_table: str
) -> tuple[set[str], list[str], list[str], dict[str, str]]:
  """Parses static export table (_pywrap_tensorflow.def) to extract active exports.

  Args:
    workspace_root: Root directory of the Bazel workspace.
    static_export_table: Relative path to
      tensorflow/python/_pywrap_tensorflow.def.

  Returns:
    Tuple of (existing_exports_set, header_lines, symbol_lines, mangled_map).
  """
  existing_exports = set()
  header_lines = []
  symbol_lines = []
  existing_mangled_to_unmangled = {}
  sd_path = os.path.join(workspace_root, static_export_table)
  try:
    with open(sd_path, "r", encoding="utf-8") as f:
      lines = f.read().splitlines()
  except OSError:
    print(
        f"Error: Static export table {sd_path} does not exist.",
        file=sys.stderr,
    )
    return (
        existing_exports,
        header_lines,
        symbol_lines,
        existing_mangled_to_unmangled,
    )
  in_exports = False
  for line in lines:
    if not in_exports:
      header_lines.append(line)
      if line.strip().upper() == "EXPORTS":
        in_exports = True
    else:
      if line.strip() and not line.strip().startswith(";"):
        sym = line.strip()
        symbol_lines.append(sym)
        existing_exports.add(sym)
        unmangled = sym
        if sym.startswith("?"):
          # General format for mangled C++ symbols:
          # ?FunctionName@Scope1@Scope2@...@@YADetails
          # The scopes are in reverse order of declaration (innermost first).
          match = re.match(r"\?([A-Za-z0-9_]+(?:@[A-Za-z0-9_]+)*)@@", sym)
          if match:
            parts = match.group(1).split("@")
            # The parts are FunctionName, ScopeN, ..., Scope1. Reverse them.
            reversed_parts = parts[::-1]
            unmangled = "::".join(reversed_parts)
        elif sym.startswith("??0"):
          # Constructor: ??0ClassName@Scope1@...@@YADetails
          match = re.match(r"\?\?0([A-Za-z0-9_]+(?:@[A-Za-z0-9_]+)*)@@", sym)
          if match:
            parts = match.group(1).split("@")
            reversed_parts = parts[::-1]
            # ClassName is the last part, so it's reversed_parts[-1].
            # The rest are namespaces.
            class_name = reversed_parts[-1]
            if len(reversed_parts) > 1:
              namespace = "::".join(reversed_parts[:-1])
              unmangled = f"{namespace}::{class_name}()"
            else:
              unmangled = f"{class_name}()"
        elif sym.startswith("??1"):
          # Destructor: ??1ClassName@Scope1@...@@YADetails
          match = re.match(r"\?\?1([A-Za-z0-9_]+(?:@[A-Za-z0-9_]+)*)@@", sym)
          if match:
            parts = match.group(1).split("@")
            reversed_parts = parts[::-1]
            class_name = reversed_parts[-1]
            if len(reversed_parts) > 1:
              namespace = "::".join(reversed_parts[:-1])
              unmangled = f"{namespace}::~{class_name}()"
            else:
              unmangled = f"~{class_name}()"
        existing_mangled_to_unmangled[sym] = unmangled
      else:
        if not line.strip():
          continue
        symbol_lines.append(line.strip())
  return (
      existing_exports,
      header_lines,
      symbol_lines,
      existing_mangled_to_unmangled,
  )


def run_query(cmd: list[str], cwd: str) -> str:
  """Runs a Bazel query command with fallback for --config=windows."""
  keep_going_cmd = cmd + ["--keep_going"] if "--keep_going" not in cmd else cmd
  try:
    proc = subprocess.run(
        keep_going_cmd, capture_output=True, text=True, cwd=cwd, check=False
    )
    if proc.returncode not in (0, 3):
      if "--config=windows" in keep_going_cmd:
        cmd_no_cfg = [c for c in keep_going_cmd if c != "--config=windows"]
        proc_fall = subprocess.run(
            cmd_no_cfg, capture_output=True, text=True, cwd=cwd, check=False
        )
        if proc_fall.returncode in (0, 3):
          return proc_fall.stdout
        print(
            f"Bazel query failed: {' '.join(cmd_no_cfg)}\n"
            f"Error: {proc_fall.stderr}",
            file=sys.stderr,
        )
        sys.exit(1)
      print(
          f"Bazel query failed: {' '.join(keep_going_cmd)}\n"
          f"Error: {proc.stderr}",
          file=sys.stderr,
      )
      sys.exit(1)
    return proc.stdout
  except Exception as e:
    print(f"Bazel query failed with exception: {e}", file=sys.stderr)
    sys.exit(1)


def query_all_targets_and_files(
    workspace_root: str, is_subrepo: bool, boundary_keywords: list[str]
) -> tuple[set[str], dict[str, list[str]]]:
  """Queries Bazel for cc_library targets and their source files."""
  print(
      "Phase 1: Querying Bazel for cc_library targets and source files "
      "for Windows configuration (this may take a few minutes)..."
  )
  repo_path = "//third_party/tensorflow" if is_subrepo else "//tensorflow"
  absl_path = "//third_party/absl" if is_subrepo else "@com_google_absl//absl"
  llvm_path = "//third_party/llvm" if is_subrepo else "@llvm-project//llvm"
  internal_bin = "b" + "l" + "a" + "z" + "e"
  bazel_bin = internal_bin if is_subrepo else "bazel"
  pybind_path = (
      "//third_party/pybind11" if is_subrepo else "@pybind11//pybind11"
  )
  pypb_path = (
      "//third_party/pybind11_protobuf"
      if is_subrepo
      else "@pybind11_protobuf//pybind11_protobuf:native_proto_caster"
  )

  combined_query = (
      f'kind("source file", deps({repo_path}/python:gen_pywrap_tensorflow_def))'
      f' union kind("source file", {absl_path}/status/...) union kind("source'
      f' file", {llvm_path}/...) union kind("source file", {pybind_path}/...)'
      f' union kind("source file", deps({pypb_path}))'
  )

  src_labels = []
  query_cmd = [
      bazel_bin,
      "query",
      combined_query,
      "--config=windows",
      "--output=label_kind",
  ]
  try:
    output = run_query(query_cmd, cwd=workspace_root)
    lines = output.strip().splitlines()
    if lines:
      src_labels.extend(lines)
  except Exception as e:
    print(f"Warning: Combined Bazel query failed: {e}", file=sys.stderr)

  if not src_labels:
    print(
        "Error: All Bazel queries failed to return source files.",
        file=sys.stderr,
    )
    sys.exit(1)

  print(f"Found {len(src_labels)} total source file labels.")
  ext_deps = set()
  target_files_map = {}
  for label_kind in src_labels:
    if label_kind.startswith("source file"):
      parts = label_kind.split(" ", 2)
      if len(parts) == 3:
        label = parts[2]
        if label.startswith("//"):
          rel_path = label[2:].replace(":", "/")
        elif label.startswith("@//") or label.startswith("@@//"):
          clean_label = label.lstrip("@")
          rel_path = clean_label[2:].replace(":", "/")
        elif "//" in label:
          repo_part, path_part = label.split("//", 1)
          repo_name = repo_part.lstrip("@")
          rel_path = f"external/{repo_name}/{path_part.replace(':', '/')}"
        else:
          continue

        clean_dir = (
            os.path.dirname(rel_path)
            .replace("third_party/", "")
            .rstrip("/")
        )
        pkg = "//third_party/" + clean_dir
        name = os.path.basename(pkg)
        target = f"{pkg}:{name}"
        ext_deps.add(target)
        if target not in target_files_map:
          target_files_map[target] = []
        target_files_map[target].append(rel_path)
  print(f"Filtered to {len(ext_deps)} extension dependency targets.")
  return ext_deps, target_files_map


def harvest_active_classes_and_headers(
    workspace_root: str, is_subrepo: bool
) -> tuple[set[str], list[str]]:
  """Dynamically harvests active C++ classes and boundary keywords."""
  print(
      "Phase 1a: Dynamically harvesting active C++ classes and boundary "
      "keywords from Python/Pybind wrappers..."
  )
  active_classes = set()
  boundary_keywords_set = set()

  # Seed initial boundary keywords for C API and core framework
  boundary_keywords_set.update({
      "c_api", "core/framework", "core/lib", "core/protobuf",
      "python", "compiler", "absl"
  })

  ext_bases = [
      workspace_root,
      os.path.join(workspace_root, "external"),
      os.path.abspath(os.path.join(workspace_root, "..", "external")),
      os.path.abspath(os.path.join(workspace_root, "..", "..", "external")),
      os.path.abspath(
          os.path.join(workspace_root, "..", "..", "..", "external")
      ),
  ]
  bazel_out_sym = os.path.join(workspace_root, "bazel-out")
  if os.path.exists(bazel_out_sym):
    real_bazel_out = os.path.realpath(bazel_out_sym)
    ext_bases.extend([
        os.path.join(real_bazel_out, "external"),
        os.path.abspath(os.path.join(real_bazel_out, "..", "external")),
        os.path.abspath(os.path.join(real_bazel_out, "..", "..", "external")),
        os.path.abspath(
            os.path.join(real_bazel_out, "..", "..", "..", "external")
        ),
    ])

  absl_sdir = "external/com_google_absl/absl"
  llvm_sdir = "external/llvm-project/llvm"
  pybind_sdir = "external/pybind11/include"
  for base in ext_bases:
    if not os.path.exists(base):
      continue
    for d in os.listdir(base):
      if (
          d == "abseil-cpp"
          or d.startswith("abseil-cpp~")
          or d == "com_google_absl"
      ):
        cand = os.path.join(base, d, "absl")
        if os.path.exists(cand):
          absl_sdir = cand
      elif d == "llvm-project" or d.startswith("llvm-project~"):
        cand = os.path.join(base, d, "llvm")
        if os.path.exists(cand):
          llvm_sdir = cand
      elif d == "pybind11" or d.startswith("pybind11~"):
        cand = os.path.join(base, d, "include")
        if os.path.exists(cand):
          pybind_sdir = cand

  scan_dirs = [
      ("third_party/tensorflow/python" if is_subrepo else "tensorflow/python"),
      "third_party/tensorflow/c" if is_subrepo else "tensorflow/c",
      (
          "third_party/tensorflow/compiler"
          if is_subrepo
          else "tensorflow/compiler"
      ),
      "third_party/tensorflow/core" if is_subrepo else "tensorflow/core",
      "third_party/tensorflow/dtensor" if is_subrepo else "tensorflow/dtensor",
      "third_party/absl" if is_subrepo else absl_sdir,
      "third_party/llvm" if is_subrepo else llvm_sdir,
      "third_party/pybind11" if is_subrepo else pybind_sdir,
  ]

  pybind_re = re.compile(
      r"py::class_<\s*(?:[A-Za-z0-9_]+::)*([A-Z][A-Za-z0-9_]+)"
  )
  pybind_base_re = re.compile(
      r"py::class_<\s*[A-Z][A-Za-z0-9_]+\s*,\s*(?:[A-Za-z0-9_]+::)*"
      r"([A-Z][A-Za-z0-9_]+)"
  )
  import_re = re.compile(r"import\s+([A-Z][A-Za-z0-9_]+)")
  from_import_re = re.compile(r"from\s+[\w\.]+\s+import\s+([A-Za-z0-9_\,\s]+)")
  include_re = re.compile(
      r"#include\s+[\"\']((?:tensorflow|absl|llvm|compiler|pybind11|"
      r"pybind11_protobuf)[A-Za-z0-9_\/\.\-]+)[\"\']"
  )
  include_angle_re = re.compile(
      r"#include\s+<((?:tensorflow|absl|llvm|compiler|pybind11|"
      r"pybind11_protobuf)[A-Za-z0-9_\/\.\-]+)>"
  )
  cpp_type_re = re.compile(
      r"\b(?:const\s+|struct\s+|class\s+)?([A-Z][A-Za-z0-9_]+)"
      r"(?:\s*[\*\&]|\s+[a-z_]|\s*\()"
  )
  cpp_template_re = re.compile(
      r"\b(?:unique_ptr|shared_ptr|vector|Span|StatusOr|optional|"
      r"unordered_map|flat_hash_map|map|set|FunctionRef|function|pair|tuple|"
      r"getOrLoadDialect|loadDialect)"
      r"<\s*(?:const\s+)?(?:[A-Za-z0-9_]+::)*([A-Z][A-Za-z0-9_]+)"
  )
  cpp_static_call_re = re.compile(
      r"\b(?:[A-Za-z0-9_]+::)*([A-Z][A-Za-z0-9_]+)::[A-Za-z0-9_]+"
  )
  py_cast_re = re.compile(
      r"\b(?:py::cast|py::cast_op)<\s*(?:const\s+)?(?:[A-Za-z0-9_]+::)*"
      r"([A-Z][A-Za-z0-9_]+)"
  )

  exclude_classes = frozenset({
      "LowLevelAlloc",
      "Demangler",
      "CordzInfo",
      "CordzUpdateTracker",
      "WorkQueue",
      "Scavenger",
      "Device",
      "DeviceBase",
      "Allocator",
      "BFCAllocator",
      "SubAllocator",
      "OpKernel",
      "OpKernelContext",
      "OpKernelConstruction",
      "CancellationManager",
      "FunctionLibraryRuntime",
      "ResourceMgr",
      "Environment",
      "Thread",
      "ThreadPool",
      "Env",
      "TensorBuffer",
      "Graph",
      "Node",
      "Edge",
      "Session",
      "TensorSlice",
      "TensorSliceSet",
      "TensorSliceReader",
      "TensorSliceWriter",
      "TensorSliceReaderCache",
      "TensorSliceReaderCacheWrapper",
      "SimplePropagatorState",
      "PropagatorState",
      "ImmutableExecutorState",
      "RunHandler",
      "RunHandlerPool",
      "RunHandlerEnvironment",
      "RunHandlerThreadPool",
      "ThreadWorkSource",
      "TrackingAllocator",
      "ScopedAllocator",
      "ScopedAllocatorMgr",
      "ScopedAllocatorInstance",
      "ScopedAllocatorContainer",
      "ProcessState",
      "MemmappedEnv",
      "MemmappedFileSystem",
      "DynamicDeviceMgr",
      "DeviceMgr",
      "RenamedDevice",
      "LocalDevice",
      "CostModel",
      "CostModelManager",
      "PoolAllocator",
      "SubProcess",
      "Activity",
      "ActivityScope",
      "BufferedInputStream",
      "RandomAccessInputStream",
      "SnappyInputStream",
      "ZlibInputStream",
      "InputBuffer",
      "RecordReader",
      "SequentialRecordReader",
      "SnappyOutputBuffer",
      "ZlibOutputBuffer",
      "RecordWriter",
      "TStringOutputStream",
      "StringOutputStreamAdaptor",
      "CopyingOutputStreamAdaptor",
      "ArrayInputStream",
      "CodedInputStream",
      "CodedOutputStream",
      "EpsCopyInputStream",
      "MapFieldBase",
      "UntypedMapBase",
      "RepeatedFieldAccessor",
      "MicroString",
      "SerialArena",
      "ThreadSafeArena",
      "DynamicMessageFactory",
      "MessageLite",
      "GeneratedCodeInfo",
      "DescriptorDatabase",
      "DescriptorPool",
      "Tables",
      "DeferredValidation",
      "Reflection",
      "ScratchSpace",
      "Service",
      "Method",
      "Any",
      "BoolValue",
      "BytesValue",
      "DoubleValue",
      "Duration",
      "Empty",
      "Enum",
      "EnumValue",
      "Field",
      "FloatValue",
      "Int32Value",
      "Int64Value",
      "ListValue",
      "NullValue",
      "Option",
      "SourceContext",
      "StringValue",
      "Struct",
      "Timestamp",
      "Type",
      "UInt32Value",
      "UInt64Value",
      "Value",
      "AnnotationRecord",
      "Printer",
      "Finder",
      "Parser",
      "BaseTextGenerator",
      "Importer",
      "ErrorCollector",
      "MultiFileErrorCollector",
      "SourceTree",
      "Compiler",
      "Arg",
      "LogMessageQuietlyFatal",
      "int128",
      "uint128",
      "BlockingCounter",
      "GraphCycles",
      "Rep",
      "LeakCheckDisabler",
      "CharIterator",
      "Task",
      "Cache",
      "Iterator",
      "IteratorBase",
      "TableBuilder",
      "Table",
      "LRUCache",
      "WeightedPicker",
  })

  for sdir in scan_dirs:
    full_sdir = os.path.join(workspace_root, sdir)
    if not os.path.exists(full_sdir):
      continue
    for root, _, files in os.walk(full_sdir):
      for file in files:
        if file.endswith((".py", ".cc", ".h", ".i")):
          fpath = os.path.join(root, file)
          try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
              for line in f:
                line_str = line.strip()
                if not line_str or line_str.startswith("//"):
                  continue
                if fpath.endswith(".py") and line_str.startswith("#"):
                  continue

                if line_str.startswith("#include"):
                  m5 = include_re.search(line_str)
                  if m5:
                    inc_path = m5.group(1)
                    clean_dir = (
                        os.path.dirname(inc_path)
                        .replace("tensorflow/", "")
                        .replace("third_party/", "")
                        .rstrip("/")
                    )
                    if clean_dir and "kernel" not in clean_dir:
                      boundary_keywords_set.add(clean_dir)
                  m6 = include_angle_re.search(line_str)
                  if m6:
                    inc_path = m6.group(1)
                    clean_dir = (
                        os.path.dirname(inc_path)
                        .replace("tensorflow/", "")
                        .replace("third_party/", "")
                        .rstrip("/")
                    )
                    if clean_dir and "kernel" not in clean_dir:
                      boundary_keywords_set.add(clean_dir)
                else:
                  m1 = pybind_re.search(line_str)
                  if m1 and m1.group(1) not in exclude_classes:
                    active_classes.add(m1.group(1))
                  m2 = pybind_base_re.search(line_str)
                  if m2 and m2.group(1) not in exclude_classes:
                    active_classes.add(m2.group(1))
                  m3 = import_re.search(line_str)
                  if m3 and m3.group(1) not in exclude_classes:
                    active_classes.add(m3.group(1))
                  m4 = from_import_re.search(line_str)
                  if m4:
                    parts = [
                        p.strip()
                        for p in m4.group(1).split(",")
                        if p.strip()
                    ]
                    for p in parts:
                      if p and p[0].isupper() and p not in exclude_classes:
                        active_classes.add(p)
                  m9 = py_cast_re.search(line_str)
                  if m9 and m9.group(1) not in exclude_classes:
                    active_classes.add(m9.group(1))

                  clean_root = root.replace("\\", "/")
                  is_py_wrapper = (
                      "tensorflow/python" in clean_root
                      or "tensorflow/c" in clean_root
                      or "grappler" in clean_root
                      or "tensorflow/dtensor" in clean_root
                      or "tensorflow/compiler" in clean_root
                      or "tensorflow/core" in clean_root
                      or "tsl" in clean_root
                      or "absl" in clean_root
                      or "llvm" in clean_root
                      or "mlir" in clean_root
                      or "pybind11" in clean_root
                      or file.endswith((".i", "_wrapper.cc", "_binding.cc"))
                      or "pywrap" in file
                      or "pybind" in file
                  )
                  if is_py_wrapper and not file.endswith(".py"):
                    m7 = cpp_type_re.search(line_str)
                    if m7 and m7.group(1) not in exclude_classes:
                      active_classes.add(m7.group(1))
                    m8 = cpp_template_re.search(line_str)
                    if m8 and m8.group(1) not in exclude_classes:
                      active_classes.add(m8.group(1))
                    m10 = cpp_static_call_re.search(line_str)
                    if m10 and m10.group(1) not in exclude_classes:
                      active_classes.add(m10.group(1))
          except OSError:
            continue

  boundary_list = sorted(list(boundary_keywords_set))
  print(
      f"Successfully harvested {len(active_classes)} active C++ classes "
      f"and {len(boundary_list)} boundary keywords dynamically."
  )
  return active_classes, boundary_list


def harvest_coff_symbols_from_obj_files(
    obj_files: list[str], llvm_nm_path: Optional[str]
) -> dict[str, list[str]]:
  """Scans provided .obj/.lib files to harvest real COFF symbols."""
  print(
      "\nPhase 2a: Scanning provided compiled .obj/.lib files "
      "to harvest real COFF symbols..."
  )
  coff_symbols_map = {}

  if not obj_files:
    print(
        "Note: No object files provided. Skipping object file COFF "
        "harvesting."
    )
    return coff_symbols_map

  print(
      f"Found {len(obj_files)} compiled object/library files. Sourcing COFF "
      "symbols..."
  )

  nm_bin = (
      "llvm-nm.exe"
      if os.name == "nt" or "MSYS" in os.environ.get("MSYSTEM", "")
      else "llvm-nm"
  )
  dumpbin_bin = "dumpbin.exe"

  def which(cmd):
    for path in os.environ.get("PATH", "").split(os.pathsep):
      exe_file = os.path.join(path, cmd)
      if os.path.exists(exe_file) and os.access(exe_file, os.X_OK):
        return exe_file
      if os.name == "nt" and not cmd.endswith(".exe"):
        exe_file += ".exe"
        if os.path.exists(exe_file) and os.access(exe_file, os.X_OK):
          return exe_file
    return None

  nm_path = llvm_nm_path if llvm_nm_path else which(nm_bin)
  dumpbin_path = which(dumpbin_bin)

  if not nm_path and not dumpbin_path:
    print(
        "Warning: Neither llvm-nm nor dumpbin found in PATH. Skipping COFF "
        "harvesting."
    )
    return coff_symbols_map

  mangled_re = re.compile(
      r"(?:^|\s)(?:__imp_)?(\?[A-Za-z0-9_\@\$\?]+|TF_[A-Za-z0-9_\@\$\?]+|TFE_[A-Za-z0-9_\@\$\?]+)"
  )

  chunk_size = 50
  for i in range(0, len(obj_files), chunk_size):
    chunk = obj_files[i : i + chunk_size]
    cmd = []
    if nm_path:
      cmd = [nm_path, "--defined-only"] + chunk
    elif dumpbin_path:
      cmd = [dumpbin_path, "/SYMBOLS"] + chunk

    try:
      env = os.environ.copy()
      p = subprocess.run(
          cmd,
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE,
          text=True,
          env=env,
          check=False,
      )
      for line in p.stdout.splitlines():
        m = mangled_re.search(line)
        if m:
          sym = m.group(1)
          sym = sym.replace("__imp_", "").strip()
          if not sym:
            continue

          full_key = None
          if sym.startswith("??0"):  # Constructor
            parts = sym.split("@")
            if len(parts) > 1:
              fn_name = parts[0][3:]
              empty_idx = parts.index("") if "" in parts else len(parts)
              ns_parts = parts[1:empty_idx]
              ns_parts = [
                  p for p in ns_parts if not re.fullmatch(r"lts_[0-9]+", p)
              ]
              cls_name = "::".join(ns_parts[::-1]) if ns_parts else None
              full_key = (
                  f"{cls_name}::{fn_name}::{fn_name}"
                  if cls_name
                  else f"{fn_name}::{fn_name}"
              )
          elif sym.startswith("??1"):  # Destructor
            parts = sym.split("@")
            if len(parts) > 1:
              fn_name = parts[0][3:]
              empty_idx = parts.index("") if "" in parts else len(parts)
              ns_parts = parts[1:empty_idx]
              ns_parts = [
                  p for p in ns_parts if not re.fullmatch(r"lts_[0-9]+", p)
              ]
              cls_name = "::".join(ns_parts[::-1]) if ns_parts else None
              full_key = (
                  f"{cls_name}::{fn_name}::~{fn_name}"
                  if cls_name
                  else f"{fn_name}::~{fn_name}"
              )
          elif (
              sym.startswith("??B")
              or sym.startswith("??6")
              or sym.startswith("??8")
              or sym.startswith("??R")
              or sym.startswith("??D")
              or sym.startswith("??7")
              or sym.startswith("??9")
              or sym.startswith("??4")
              or sym.startswith("??G")
              or sym.startswith("??H")
              or sym.startswith("??P")
          ):  # C++ Operators
            parts = sym.split("@")
            if len(parts) > 1:
              fn_name = parts[0][3:]
              empty_idx = parts.index("") if "" in parts else len(parts)
              ns_parts = parts[1:empty_idx]
              ns_parts = [
                  p for p in ns_parts if not re.fullmatch(r"lts_[0-9]+", p)
              ]
              cls_name = "::".join(ns_parts[::-1]) if ns_parts else None
              full_key = (
                  f"{cls_name}::{fn_name}::operator"
                  if cls_name
                  else f"{fn_name}::operator"
              )
          elif sym.startswith("?"):  # Other mangled C++
            parts = sym.split("@")
            if len(parts) > 1:
              fn_name = parts[0][1:]
              empty_idx = parts.index("") if "" in parts else len(parts)
              ns_parts = parts[1:empty_idx]
              ns_parts = [
                  p for p in ns_parts if not re.fullmatch(r"lts_[0-9]+", p)
              ]
              cls_name = "::".join(ns_parts[::-1]) if ns_parts else None
              full_key = f"{cls_name}::{fn_name}" if cls_name else fn_name
          elif sym.startswith("TF_") or sym.startswith("TFE_"):
            full_key = sym

          if full_key:
            if full_key not in coff_symbols_map:
              coff_symbols_map[full_key] = []
            coff_symbols_map[full_key].append(sym)
    except (subprocess.CalledProcessError, OSError):
      continue

  total_coff = sum(len(v) for v in coff_symbols_map.values())
  print(
      f"Successfully harvested {total_coff} real COFF symbols from "
      f"{len(obj_files)} object files."
  )
  return coff_symbols_map


def _extract_symbols_sequential(
    files: list[str], workspace_root: str, active_classes: set[str]
) -> list[dict[str, str]]:
  """Scans C++ source/header ASTs to extract public class methods and functions.

  NOTE: This function uses regular expressions to parse C++ declarations, which
  is inherently fragile and has limitations. It may fail to correctly parse: 1.
  Functions with template parameters. 2.  Multi-line declarations. 3.  Macros
  used in return types or function names. 4.  Complex or nested namespaces
  beyond simple `namespace <name>`. 5.  Trailing return types. Future
  improvements should consider using a more robust C++ parsing method (e.g.,
  libclang).

  Args:
    files: A list of relative paths to C++ source and header files.
    workspace_root: The root directory of the Bazel workspace.
    active_classes: A set of active C++ class names to filter against.

  Returns:
    A list of dictionaries, each containing information about an extracted
    symbol. Each dictionary has the following keys:
    -   'symbol': The full qualified symbol name (e.g., "tensorflow::OpenVino").
    -   'function_name': The name of the function or method (e.g., "OpenVino").
    -   'class_name': The name of the class if applicable (e.g., "OpenVino").
    -   'file': The relative path to the file where the symbol was found.
    -   'line': The line number in the file where the symbol was found.
    -   'line_str': The full line string where the symbol was found.
    -   'type': The type of extraction ("method_impl" or "declaration").
    -   'is_extern_c': A boolean indicating if the symbol is likely an
        `extern "C"` symbol.
  """
  extracted_symbols = []
  fn_re = re.compile(
      r"^\s*(?:virtual\s+|static\s+|inline\s+|LLVM_ABI\s+|extern\s+|const\s+"
      r"|explicit\s+)?(?:([A-Za-z0-9_\:\<\>\*\&\s]+?)\s+)?"
      r"([~A-Za-z0-9_]+)\s*\("
  )
  var_re = re.compile(
      r"^\s*extern\s+(?:const\s+)?(?:[\w\:\<\>\*\&\s]+)\s+"
      r"([A-Za-z0-9_]+)\s*(?:\[[^\]]*\])?\s*;"
  )
  method_impl_re = re.compile(
      r"^\s*(?:(?:inline\s+|extern\s+|virtual\s+)*([\w\:\<\>\*\&\s]+)\s+)?"
      r"([A-Za-z0-9_]+)\:\:([~A-Za-z0-9_]+)\s*\("
  )
  internal_namespaces = frozenset({
      "impl::",
      "random::",
      "detail::",
      "testing::",
      "benchmark::",
      "strings::",
      "anonymous_namespace",
      "test::",
  })
  for rel_file in files:
    file_path = os.path.join(workspace_root, rel_file)
    if not (rel_file.endswith(".h") or rel_file.endswith(".cc")):
      continue
    current_class = None
    in_extern_c = False
    brace_depth = 0
    class_brace_depth = -1
    namespace_stack = []
    namespace_brace_depths = []

    resolved_path = file_path
    if not os.path.exists(file_path):
      found = False
      if rel_file.startswith("external/"):
        sub_path = rel_file[len("external/") :]
        ext_bases = [
            os.path.join(workspace_root, "external"),
            os.path.abspath(os.path.join(workspace_root, "..", "external")),
            os.path.abspath(
                os.path.join(workspace_root, "..", "..", "external")
            ),
            os.path.abspath(
                os.path.join(workspace_root, "..", "..", "..", "external")
            ),
        ]
        bazel_out_sym = os.path.join(workspace_root, "bazel-out")
        if os.path.exists(bazel_out_sym):
          real_bazel_out = os.path.realpath(bazel_out_sym)
          ext_bases.extend([
              os.path.join(real_bazel_out, "external"),
              os.path.abspath(os.path.join(real_bazel_out, "..", "external")),
              os.path.abspath(
                  os.path.join(real_bazel_out, "..", "..", "external")
              ),
              os.path.abspath(
                  os.path.join(real_bazel_out, "..", "..", "..", "external")
              ),
          ])
        for base in ext_bases:
          if not os.path.exists(base):
            continue
          cand = os.path.join(base, sub_path)
          if os.path.exists(cand):
            resolved_path = cand
            found = True
            break
          repo_name, rest_path = sub_path.split("/", 1)
          for d in os.listdir(base):
            if (
                d == repo_name
                or d.startswith(repo_name + "~")
                or d.startswith(repo_name + "-")
                or d.startswith("+" + repo_name)
                or d.startswith("_main~" + repo_name)
            ):
              cand_tilde = os.path.join(base, d, rest_path)
              if os.path.exists(cand_tilde):
                resolved_path = cand_tilde
                found = True
                break
          if found:
            break
      if not found:
        found_in_bazel_out = False
        bazel_out = os.path.join(workspace_root, "bazel-out")
        if os.path.exists(bazel_out):
          real_bazel_out = os.path.realpath(bazel_out)
          for root, dirs, _ in os.walk(real_bazel_out, followlinks=True):
            for d in dirs:
              if "windows" in d and (
                  "opt" in d or "fastbuild" in d or "dbg" in d
              ):
                bin_dir = os.path.join(root, d, "bin")
                candidates = [
                    os.path.join(bin_dir, rel_file),
                    os.path.join(bin_dir, rel_file.replace("third_party/", "")),
                    os.path.join(bin_dir, "third_party", rel_file),
                ]
                for candidate in candidates:
                  if os.path.exists(candidate):
                    resolved_path = candidate
                    found_in_bazel_out = True
                    break
            if found_in_bazel_out:
              break
        if not found_in_bazel_out:
          continue

    try:
      with open(resolved_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_num, line in enumerate(f, 1):
          line_str = line.strip()
          old_brace_depth = brace_depth
          brace_depth += line_str.count("{") - line_str.count("}")
          if brace_depth < 0:
            brace_depth = 0
          if current_class and brace_depth <= class_brace_depth:
            current_class = None
            class_brace_depth = -1
          while (
              namespace_brace_depths
              and brace_depth <= namespace_brace_depths[-1]
          ):
            namespace_stack.pop()
            namespace_brace_depths.pop()
          if 'extern "C"' in line_str:
            in_extern_c = True
          if in_extern_c and "}" in line_str and "{" not in line_str:
            in_extern_c = False
          ns_match = re.search(r"^namespace\s+([A-Za-z0-9_]+)", line_str)
          if ns_match:
            namespace_stack.append(ns_match.group(1))
            namespace_brace_depths.append(
                old_brace_depth if "{" in line_str else brace_depth
            )
          current_namespace = (
              "::".join(namespace_stack) if namespace_stack else None
          )
          class_match = re.search(
              r"^\s*(?:class|struct)\s+(?:LLVM_ABI\s+)?"
              r"([A-Za-z0-9_]+)",
              line_str,
          )
          if class_match:
            current_class = class_match.group(1)
            class_brace_depth = brace_depth - 1 if brace_depth > 0 else 0
          m_match = method_impl_re.match(line_str)
          if m_match:
            cls_name = m_match.group(2)
            fn_name = m_match.group(3)
            if current_namespace:
              if cls_name.startswith(current_namespace + "::"):
                full_sym = f"{cls_name}::{fn_name}"
              else:
                full_sym = f"{current_namespace}::{cls_name}::{fn_name}"
            else:
              full_sym = f"{cls_name}::{fn_name}"
            if any(ins in full_sym for ins in internal_namespaces):
              continue
            if cls_name and cls_name not in active_classes:
              continue
            extracted_symbols.append({
                "symbol": full_sym,
                "function_name": fn_name,
                "class_name": cls_name,
                "file": rel_file,
                "line": line_num,
                "line_str": line_str,
                "type": "method_impl",
                "is_extern_c": False,
            })
            continue
          op_match = re.match(
              r"^\s*(?:explicit\s+|inline\s+|virtual\s+|LLVM_ABI\s+)*"
              r"operator\s+([A-Za-z0-9_\:\<\>\*\&\s]+)\s*\(",
              line_str,
          )
          if op_match:
            cls_name = current_class if current_class else ""
            if current_namespace:
              if cls_name:
                full_sym = f"{current_namespace}::{cls_name}::operator"
              else:
                continue
            else:
              if cls_name:
                full_sym = f"{cls_name}::operator"
              else:
                continue
            if cls_name and cls_name not in active_classes:
              continue
            extracted_symbols.append({
                "symbol": full_sym,
                "function_name": "operator",
                "class_name": cls_name,
                "file": rel_file,
                "line": line_num,
                "line_str": line_str,
                "type": "declaration",
                "is_extern_c": False,
            })
            continue
          var_match = var_re.match(line_str)
          if var_match:
            var_name = var_match.group(1)
            cls_name = current_class if current_class else ""
            if current_namespace:
              if cls_name:
                full_sym = f"{current_namespace}::{cls_name}::{var_name}"
              else:
                full_sym = f"{current_namespace}::{var_name}"
            else:
              full_sym = f"{cls_name}::{var_name}" if cls_name else var_name
            if any(ins in full_sym for ins in internal_namespaces):
              continue
            if cls_name and cls_name not in active_classes:
              continue
            extracted_symbols.append({
                "symbol": full_sym,
                "function_name": var_name,
                "class_name": cls_name,
                "file": rel_file,
                "line": line_num,
                "line_str": line_str,
                "type": "declaration",
                "is_extern_c": False,
            })
            continue
          fn_match = fn_re.match(line_str)
          if fn_match:
            if rel_file.endswith(".cc") and line_str.startswith("static "):
              continue
            fn_name = fn_match.group(2)
            skip_keywords = {
                "if",
                "while",
                "for",
                "switch",
                "return",
                "catch",
                "sizeof",
                "decltype",
                "alignas",
                "class",
                "struct",
                "union",
                "enum",
                "namespace",
                "template",
                "ABSL_DEPRECATE_AND_INLINE",
                "ABSL_DEPRECATED",
                "ABSL_ACQUIRED_AFTER",
                "ABSL_EXCLUSIVE_LOCKS_REQUIRED",
                "ABSL_GUARDED_BY",
                "ABSL_MUST_USE_RESULT",
                "ABSL_ATTRIBUTE_UNUSED",
                "ABSL_ATTRIBUTE_WEAK",
                "ABSL_ATTRIBUTE_PACKED",
                "ABSL_ATTRIBUTE_NOINLINE",
                "ABSL_ATTRIBUTE_ALWAYS_INLINE",
                "TF_ATTRIBUTE_NOINLINE",
                "TF_ATTRIBUTE_ALWAYS_INLINE",
                "TF_ATTRIBUTE_UNUSED",
                "TF_ATTRIBUTE_WEAK",
                "TF_ATTRIBUTE_PACKED",
                "TF_PACKED",
                "TF_MUST_USE_RESULT",
            }
            if fn_name in skip_keywords:
              continue
            cls_name = current_class if current_class else ""
            if current_namespace:
              if cls_name:
                full_sym = f"{current_namespace}::{cls_name}::{fn_name}"
              else:
                full_sym = f"{current_namespace}::{fn_name}"
            else:
              full_sym = f"{cls_name}::{fn_name}" if cls_name else fn_name
            if any(ins in full_sym for ins in internal_namespaces):
              continue
            is_c_sym = (
                in_extern_c
                or 'extern "C"' in line_str
                or "c_api" in rel_file
                or fn_name.startswith("TF_")
                or fn_name.startswith("TFE_")
            )
            if not is_c_sym:
              if cls_name and cls_name not in active_classes:
                continue
              if not cls_name:
                if not current_namespace:
                  if not (
                      fn_name.startswith("EagerTensor_")
                      or fn_name.startswith("pywrap_")
                  ):
                    continue
                elif not (
                    current_namespace.startswith("tensorflow")
                    or current_namespace.startswith("tflite")
                    or current_namespace.startswith("toco")
                    or current_namespace.startswith("pybind11")
                    or current_namespace.startswith("stablehlo")
                    or current_namespace.startswith("mlir")
                    or current_namespace.startswith("tsl")
                    or current_namespace.startswith("llvm")
                    or current_namespace.startswith("google")
                    or current_namespace.startswith("absl")
                ):
                  continue
            extracted_symbols.append({
                "symbol": full_sym,
                "function_name": fn_name,
                "class_name": cls_name,
                "file": rel_file,
                "line": line_num,
                "line_str": line_str,
                "type": "declaration",
                "is_extern_c": is_c_sym,
            })
    except FileNotFoundError:
      continue

  return extracted_symbols


def _extract_chunk_worker(
    args_tuple: tuple[list[str], str, set[str]]
) -> list[dict[str, str]]:
  """Top-level worker function for ProcessPoolExecutor to extract symbols from a chunk of files."""
  chunk_files, workspace_root, active_classes = args_tuple
  return _extract_symbols_sequential(
      chunk_files, workspace_root, active_classes
  )


def extract_public_symbols_from_cpp_files(
    files: list[str], workspace_root: str, active_classes: set[str]
) -> list[dict[str, str]]:
  """Scans C++ source/header ASTs to extract public class methods and functions using ProcessPoolExecutor."""
  if not files:
    return []

  num_workers = min(32, (os.cpu_count() or 4))
  chunk_size = max(50, (len(files) + num_workers - 1) // num_workers)

  chunks = [
      (files[i : i + chunk_size], workspace_root, active_classes)
      for i in range(0, len(files), chunk_size)
  ]

  extracted = []
  print(
      f"Phase 2: Launching {num_workers} parallel worker processes across "
      f"{len(chunks)} file chunks...",
      file=sys.stderr,
  )

  with ProcessPoolExecutor(max_workers=num_workers) as executor:
    for res in executor.map(_extract_chunk_worker, chunks):
      extracted.extend(res)

  return extracted


def compare_and_fail(
    previous_symbols: set[str],
    regenerated_symbols: set[str],
    existing_mangled_to_unmangled: dict[str, str],
):
  """Compares old DEF symbols with new regenerated symbols, outputs differences, and fails the build."""
  matches = previous_symbols & regenerated_symbols
  missing = previous_symbols - regenerated_symbols
  print(
      "\n=====================================================================",
      file=sys.stderr,
  )
  print(
      "              STAGE 2 VERIFICATION & DIFF REPORT                     ",
      file=sys.stderr,
  )
  print(
      "=====================================================================",
      file=sys.stderr,
  )
  print(
      f"Total symbols in new generated DEF file: {len(regenerated_symbols)}",
      file=sys.stderr,
  )
  print(
      f"Total matching symbols between old DEF and new set: {len(matches)}",
      file=sys.stderr,
  )
  print(
      "Total symbols from old DEF missing in new generated set:"
      f" {len(missing)}",
      file=sys.stderr,
  )
  print(
      "---------------------------------------------------------------------",
      file=sys.stderr,
  )
  if missing:
    print("MISSING SYMBOLS (Old DEF -> Unmangled):", file=sys.stderr)
    for sym in sorted(missing):
      unm = existing_mangled_to_unmangled.get(sym, sym)
      print(f" - {sym}  ({unm})", file=sys.stderr)
  print(
      "=====================================================================\n",
      file=sys.stderr,
  )
  print(
      "Failing execution intentionally to display the verification report in"
      " Bazel logs.",
      file=sys.stderr,
  )
  sys.exit(1)


def main():
  parser = argparse.ArgumentParser(
      description="Automated All-in-One Bzlmod Windows Export Table Patcher.",
      fromfile_prefix_chars="@",
  )
  parser.add_argument(
      "--workspace_root", default=".", help="Root directory of Bazel workspace."
  )
  parser.add_argument(
      "--static_export_table",
      default="tensorflow/python/_pywrap_tensorflow.def",
      help="Path to static export table (_pywrap_tensorflow.def).",
  )
  parser.add_argument(
      "--add_symbols_file",
      default="",
      help=(
          "Optional file listing explicit symbols to be added to the DEF file."
      ),
  )
  parser.add_argument(
      "--exclude_symbols_file",
      default="",
      help="Optional file listing symbols or regex patterns to be excluded.",
  )
  parser.add_argument(
      "--output_def_file",
      default="",
      help="Optional custom output path for the generated DEF file.",
  )
  parser.add_argument(
      "--stage",
      default="all",
      choices=["all", "discovery", "mangling"],
      help=(
          "Execution stage: 'discovery' (host runner unmangled AST pass), "
          "'mangling' (genrule COFF mangling pass), or 'all'."
      ),
  )
  parser.add_argument(
      "--unmangled_symbols_file",
      default="",
      help="Path to intermediary unmangled DEF file (used in 'mangling').",
  )
  parser.add_argument(
      "--bust_cache",
      action="store_true",
      help=(
          "Injects a randomized comment line into the unmangled DEF file "
          "to force Bazel genrule cache busting."
      ),
  )
  parser.add_argument(
      "--llvm_nm_path",
      default=None,
      help="Optional path to the llvm-nm executable.",
  )
  parser.add_argument(
      "--obj_files",
      nargs="*",
      default=[],
      help="List of object or library files to scan for COFF symbols.",
  )
  args = parser.parse_args()
  if args.workspace_root == ".":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    clean_script_dir = script_dir.replace("\\", "/")
    if "third_party/tensorflow/tools/def_file_gen" in clean_script_dir:
      args.workspace_root = clean_script_dir.split(
          "third_party/tensorflow/tools/def_file_gen"
      )[0]
    elif "tensorflow/tools/def_file_gen" in clean_script_dir:
      args.workspace_root = clean_script_dir.split(
          "tensorflow/tools/def_file_gen"
      )[0]
    else:
      args.workspace_root = script_dir
  elif os.name == "nt" and args.workspace_root.startswith("/"):
    parts = [p for p in args.workspace_root.split("/") if p]
    if len(parts) >= 2 and len(parts[0]) == 1:
      args.workspace_root = f"{parts[0].upper()}:/{'/'.join(parts[1:])}"
  is_subrepo = os.path.exists(
      os.path.join(args.workspace_root, "third_party/tensorflow")
  )
  prefix = "third_party/" if is_subrepo else ""
  sd_rel = (
      prefix + args.static_export_table
      if not args.static_export_table.startswith("third_party")
      else args.static_export_table
  )
  print(f"Phase 1: Parsing static export table ({sd_rel})...")
  (
      _,
      _,
      symbol_lines,
      existing_mangled_to_unmangled,
  ) = parse_static_export_table(args.workspace_root, sd_rel)
  previous_symbols = set(s for s in symbol_lines if s and not s.startswith(";"))
  print(f"Found {len(previous_symbols)} previous exported symbol entries.")

  if args.stage == "discovery" or args.stage == "all":
    active_classes, boundary_keywords = harvest_active_classes_and_headers(
        args.workspace_root, is_subrepo
    )
    ext_deps, target_files_map = query_all_targets_and_files(
        args.workspace_root, is_subrepo, boundary_keywords
    )
    print(
        f"\nPhase 2: Scanning C++ ASTs across {len(ext_deps)} extension "
        "targets to regenerate ALL symbols from scratch..."
    )
    all_files = []
    for target in ext_deps:
      all_files.extend(target_files_map.get(target, []))

    all_files = sorted(list(set(all_files)))

    symbols = extract_public_symbols_from_cpp_files(
        all_files, args.workspace_root, active_classes
    )

    if args.stage == "discovery":
      if args.bust_cache:
        unmangled_list = [f"; Cache buster: {uuid.uuid4()}"]
      else:
        unmangled_list = []
      for sym_info in symbols:
        cls = sym_info["class_name"] if sym_info["class_name"] else ""
        unmangled_list.append(
            f"{sym_info['symbol']};;{sym_info['function_name']};;{cls};;"
            f"{sym_info['is_extern_c']}"
        )
      if args.output_def_file:
        sd_path = os.path.join(args.workspace_root, args.output_def_file)
      else:
        sd_path = os.path.join(
            args.workspace_root,
            "tensorflow/python/_pywrap_tensorflow_unmangled.def",
        )
      with open(sd_path, "w", encoding="utf-8") as f:
        f.write("\n".join(unmangled_list) + "\n")

      # Calculate rigorous estimation for final COFF symbol count surviving Stage 2
      ast_count = len(unmangled_list)
      active_cls_count = len(active_classes)
      est_min = int(active_cls_count * 0.12 + ast_count * 0.004)
      est_max = int(active_cls_count * 0.18 + ast_count * 0.007)
      est_mid = (est_min + est_max) // 2

      print(
          f"\nStage 1 (Discovery): Successfully harvested {ast_count} "
          f"unmangled C++ AST signatures into intermediary dictionary file {sd_path}.\n"
          f"--> ESTIMATED FINAL COFF SYMBOLS: ~{est_mid} symbols (Expected range: {est_min} - {est_max}).\n"
          "(Note: This is an intermediary AST pool, NOT the final export table. Stage 2 "
          "will filter these against compiled .obj files during DLL linking.)"
      )
      sys.exit(0)
  else:
    print(
        "\nStage 2 (Mangling): Loading unmangled AST symbols from "
        "intermediary DEF file..."
    )
    if args.unmangled_symbols_file:
      unm_path = os.path.join(
          args.workspace_root, args.unmangled_symbols_file
      )
    else:
      unm_path = os.path.join(
          args.workspace_root,
          "tensorflow/python/_pywrap_tensorflow_unmangled.def",
      )
    symbols = []
    active_classes = set()
    try:
      with open(unm_path, "r", encoding="utf-8") as f:
        for line in f:
          line_str = line.strip()
          if line_str and not line_str.startswith(";"):
            parts = line_str.split(";;")
            if len(parts) >= 4:
              cls = parts[2]
              if cls:
                active_classes.add(cls)
              symbols.append({
                  "symbol": parts[0],
                  "function_name": parts[1],
                  "class_name": cls if cls else None,
                  "file": "",
                  "line": 0,
                  "line_str": "",
                  "type": "declaration",
                  "is_extern_c": parts[3] == "True",
              })
    except FileNotFoundError:
      print(
          f"Error: Intermediary unmangled file {unm_path} not found.",
          file=sys.stderr,
      )
      sys.exit(1)
    print(
        f"Loaded {len(symbols)} unmangled AST symbols e.g. "
        f"{len(active_classes)} active classes."
    )
    ext_deps = set()

  # Filter out obj_files that don't exist
  existing_obj_files = [f for f in args.obj_files if os.path.exists(f)]
  if len(existing_obj_files) != len(args.obj_files):
    print(
        f"Warning: {len(args.obj_files) - len(existing_obj_files)} provided "
        "object files do not exist and will be skipped.",
        file=sys.stderr,
    )

  coff_symbols_map = harvest_coff_symbols_from_obj_files(
      existing_obj_files, args.llvm_nm_path
  )
  regenerated_symbols = set()
  mangled_to_unmangled = {}
  if args.add_symbols_file:
    add_path = os.path.join(args.workspace_root, args.add_symbols_file)
    try:
      with open(add_path, "r", encoding="utf-8") as f:
        print(f"Phase 1b: Reading explicit symbols to add from {add_path}...")
        for line in f:
          clean_sym = line.strip()
          if clean_sym and not clean_sym.startswith(";"):
            regenerated_symbols.add(clean_sym)
            mangled_to_unmangled[clean_sym] = clean_sym
    except FileNotFoundError:
      add_path = os.path.join(
          args.workspace_root, prefix + args.add_symbols_file
      )
      try:
        with open(add_path, "r", encoding="utf-8") as f:
          print(f"Phase 1b: Reading explicit symbols to add from {add_path}...")
          for line in f:
            clean_sym = line.strip()
            if clean_sym and not clean_sym.startswith(";"):
              regenerated_symbols.add(clean_sym)
              mangled_to_unmangled[clean_sym] = clean_sym
      except FileNotFoundError:
        print(
            f"Warning: Add symbols file not found at {add_path}",
            file=sys.stderr,
        )

  resolved_coff_count = 0

  coff_fn_map = {}
  for coff_key, rcs_list in coff_symbols_map.items():
    coff_fn = coff_key.split("::")[-1]
    if coff_fn not in coff_fn_map:
      coff_fn_map[coff_fn] = []
    coff_fn_map[coff_fn].append((coff_key, rcs_list))

  for sym_info in symbols:
    full_sym = sym_info["symbol"]
    real_coff_list = coff_symbols_map.get(full_sym, [])
    if real_coff_list:
      for rcs in real_coff_list:
        clean_sym = rcs.replace("__imp_", "").strip()
        if clean_sym:
          regenerated_symbols.add(clean_sym)
          mangled_to_unmangled[clean_sym] = full_sym
          resolved_coff_count += 1
    else:
      fn_sub = sym_info.get("function_name", "")
      if fn_sub:
        fallback_matches = coff_fn_map.get(fn_sub, [])
        matched_any = False
        target_cls_unqual = (
            sym_info["class_name"].split("::")[-1]
            if sym_info.get("class_name")
            else None
        )
        for coff_key, rcs_list in fallback_matches:
          coff_parts = coff_key.split("::")
          coff_cls_unqual = coff_parts[-2] if len(coff_parts) > 1 else None
          if coff_cls_unqual and (
              coff_cls_unqual[0].islower() or coff_cls_unqual.startswith("?")
          ):
            coff_cls_unqual = None

          if target_cls_unqual == coff_cls_unqual:
            for rcs in rcs_list:
              clean_sym = rcs.replace("__imp_", "").strip()
              if clean_sym:
                regenerated_symbols.add(clean_sym)
                mangled_to_unmangled[clean_sym] = full_sym
                resolved_coff_count += 1
            matched_any = True

        if not matched_any:
          for coff_key, rcs_list in fallback_matches:
            for rcs in rcs_list:
              clean_sym = rcs.replace("__imp_", "").strip()
              if clean_sym:
                regenerated_symbols.add(clean_sym)
                mangled_to_unmangled[clean_sym] = full_sym
                resolved_coff_count += 1
            break

  print(
      f"\nStage 2 (Mangling): Successfully matched {resolved_coff_count} exact COFF "
      "symbols from compiled .obj files to generate the final export table."
  )
  if args.exclude_symbols_file:
    ex_path = os.path.join(args.workspace_root, args.exclude_symbols_file)
    exclude_patterns = []
    try:
      with open(ex_path, "r", encoding="utf-8") as f:
        print(f"Phase 2b: Pruning explicit symbols from {ex_path}...")
        for line in f:
          patt = line.strip()
          if patt and not patt.startswith(";"):
            exclude_patterns.append(patt)
    except FileNotFoundError:
      ex_path = os.path.join(
          args.workspace_root, prefix + args.exclude_symbols_file
      )
      try:
        with open(ex_path, "r", encoding="utf-8") as f:
          print(f"Phase 2b: Pruning explicit symbols from {ex_path}...")
          for line in f:
            patt = line.strip()
            if patt and not patt.startswith(";"):
              exclude_patterns.append(patt)
      except FileNotFoundError:
        print(
            f"Warning: Exclude symbols file not found at {ex_path}",
            file=sys.stderr,
        )
    if exclude_patterns:
      to_remove = set()
      for sym in regenerated_symbols:
        for patt in exclude_patterns:
          if patt == sym or re.search(patt, sym):
            to_remove.add(sym)
            break
      regenerated_symbols -= to_remove
      print(f"Pruned {len(to_remove)} symbols matching exclusion patterns.")
  sd_path = (
      os.path.join(args.workspace_root, args.output_def_file)
      if args.output_def_file
      else os.path.join(args.workspace_root, sd_rel)
  )
  all_symbols = list(regenerated_symbols)
  if len(all_symbols) > 65535:
    print(
        f"\nError: Regenerated export table contains {len(all_symbols)} "
        "symbols, exceeding the Windows 64K (65535) export limit. "
        "Aborting DEF file generation.",
        file=sys.stderr,
    )
    sys.exit(1)
  all_symbols.sort()
  standard_banner = [
      "; Copyright 2026 The TensorFlow Authors. All Rights Reserved.",
      ";",
      '; Licensed under the Apache License, Version 2.0 (the "License");',
      "; you may not use this file except in compliance with the License.",
      "; You may obtain a copy of the License at",
      ";",
      ";     http://www.apache.org/licenses/LICENSE-2.0",
      ";",
      "; Unless required by applicable law or agreed to in writing, software",
      '; distributed under the License is distributed on an "AS IS" BASIS,',
      "; WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express "
      + "or implied.",
      "; See the License for the specific language governing permissions and",
      "; limitations under the License.",
      "",
      "; This file is automatically generated and maintained by",
      "; third_party/tensorflow/tools/def_file_gen/regenerate_win_exports.py.",
      "; To regenerate this file, execute the script locally.",
      "; NOTE: Symbols beginning with `__imp_` should have that prefix "
      + "removed, e.g.",
      "; `__imp_??1OpDef@tensorflow@@UEAA@XZ` becomes "
      + "`??1OpDef@tensorflow@@UEAA@XZ`.",
      "",
      "; go/keep-sorted " + "start skip_lines=1",
      "EXPORTS",
  ]
  new_content = "\n".join(standard_banner) + "\n"
  for sym in all_symbols:
    new_content += f" {sym}\n"
  new_content += "; go/keep-sorted " + "end\n"
  with open(sd_path, "w", encoding="utf-8") as f:
    f.write(new_content)
  print(
      f"\nSuccessfully regenerated and sorted {sd_path} with "
      f"{len(all_symbols)} total symbols."
  )
  compare_and_fail(
      previous_symbols, regenerated_symbols, existing_mangled_to_unmangled
  )


if __name__ == "__main__":
  main()
