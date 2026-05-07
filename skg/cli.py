"""skg CLI entry point.

Commands:
  skg route "<task>"              Route a task through all stages and print the result.
  skg node add <manifest.yaml>    Register a node from a manifest file.
  skg node list                   List all active nodes.
  skg node inspect <id>           Print full node details.
  skg node retire <id>            Mark a node stale.
  skg attestation log <node_id>   Print all attestation records for a node.
  skg index rebuild               Rebuild SQLite FTS and Qdrant indexes from the node store.
  skg doctor                      Check the node store for schema errors, stale nodes, etc.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="skg",
        description="Skill Knowledge Graph — capability-governed procedure reuse for LLM agents.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # skg route
    p_route = sub.add_parser("route", help="Route a task through the SKG.")
    p_route.add_argument("task", help="Natural-language task description.")
    p_route.add_argument("--context", "-c", default="{}", help="JSON context dict.")
    p_route.add_argument("--json", action="store_true", dest="json_out", help="Output as JSON.")

    # skg node
    p_node = sub.add_parser("node", help="Manage nodes.")
    node_sub = p_node.add_subparsers(dest="node_cmd", required=True)

    p_add = node_sub.add_parser("add", help="Register a node from a manifest YAML file.")
    p_add.add_argument("manifest", help="Path to manifest.yaml.")

    p_list = node_sub.add_parser("list", help="List all active nodes.")
    p_list.add_argument("--all", action="store_true", dest="list_all", help="Include stale nodes.")

    p_inspect = node_sub.add_parser("inspect", help="Inspect a node.")
    p_inspect.add_argument("node_id")

    p_retire = node_sub.add_parser("retire", help="Mark a node stale.")
    p_retire.add_argument("node_id")

    p_promote = node_sub.add_parser("promote", help="Promote a candidate node to active.")
    p_promote.add_argument("node_id")

    # skg attestation
    p_attest = sub.add_parser("attestation", help="Manage attestation records.")
    attest_sub = p_attest.add_subparsers(dest="attest_cmd", required=True)
    p_attest_log = attest_sub.add_parser("log", help="Print attestation records for a node.")
    p_attest_log.add_argument("node_id")

    # skg index
    p_index = sub.add_parser("index", help="Manage SKG indexes.")
    index_sub = p_index.add_subparsers(dest="index_cmd", required=True)
    index_sub.add_parser("rebuild", help="Rebuild FTS and vector indexes.")

    # skg doctor
    sub.add_parser("doctor", help="Check node store health.")

    args = parser.parse_args()

    if args.command == "route":
        cmd_route(args)
    elif args.command == "node":
        cmd_node(args)
    elif args.command == "attestation":
        cmd_attestation(args)
    elif args.command == "index":
        cmd_index(args)
    elif args.command == "doctor":
        cmd_doctor(args)


# ---- Commands ----------------------------------------------------------------

def cmd_route(args) -> None:
    from skg.graph import SKG
    skg = SKG()
    try:
        context = json.loads(args.context)
    except json.JSONDecodeError as e:
        print(f"Error: --context is not valid JSON: {e}", file=sys.stderr)
        sys.exit(1)

    result = skg.route(args.task, context)

    if args.json_out:
        print(json.dumps({
            "hit":    result.hit,
            "stage":  result.stage,
            "node_id": result.node.id if result.node else None,
            "reason": result.reason,
            "grant":  result.grant.to_dict() if result.grant else None,
        }, indent=2))
        return

    if result.hit:
        node = result.node
        print(f"HIT  [{result.stage}]  {node.id}")
        print(f"     header: {node.manifest.header[:80]}")
        if result.grant:
            granted = [c.effect for c in result.grant.granted]
            denied  = result.grant.denied
            print(f"     granted: {granted}")
            if denied:
                print(f"     denied:  {denied}")
    else:
        print(f"MISS  {result.reason}")


def cmd_node(args) -> None:
    from skg.graph import SKG
    from skg.node import Node, Manifest
    skg = SKG()

    if args.node_cmd == "add":
        path = Path(args.manifest)
        if not path.exists():
            print(f"Error: {path} not found.", file=sys.stderr)
            sys.exit(1)
        try:
            import yaml
            data = yaml.safe_load(path.read_text())
        except Exception as e:
            print(f"Error parsing manifest: {e}", file=sys.stderr)
            sys.exit(1)
        node = Node.from_dict(data)
        skg.add_node(node)
        print(f"Added node: {node.id} ({node.status})")

    elif args.node_cmd == "list":
        store = skg._store
        nodes = store.list_all() if args.list_all else store.list_active()
        if not nodes:
            print("No nodes.")
            return
        for n in nodes:
            print(f"  {n.id:<40} {n.status:<10} {n.manifest.header[:50]}")

    elif args.node_cmd == "inspect":
        node = skg.get_node(args.node_id)
        if not node:
            print(f"Node '{args.node_id}' not found.", file=sys.stderr)
            sys.exit(1)
        print(json.dumps(node.to_dict(), indent=2))

    elif args.node_cmd == "retire":
        skg.retire(args.node_id)
        print(f"Retired: {args.node_id}")

    elif args.node_cmd == "promote":
        skg.promote(args.node_id)
        print(f"Promoted: {args.node_id}")


def cmd_attestation(args) -> None:
    from skg.graph import SKG
    skg = SKG()
    records = skg.attestations(args.node_id)
    if not records:
        print(f"No attestation records for '{args.node_id}'.")
        return
    for r in records:
        print(json.dumps(r.to_dict(), indent=2))


def cmd_index(args) -> None:
    if args.index_cmd == "rebuild":
        from skg.graph import SKG
        from skg.index import VectorIndex
        skg = SKG()
        nodes = skg._store.list_active()
        idx = VectorIndex()
        node_dicts = [
            {"node_id": n.id, "task_type": n.manifest.task_type, "header": n.manifest.header}
            for n in nodes
        ]
        count = idx.rebuild(node_dicts)
        print(f"Rebuilt vector index: {count} nodes indexed.")
        print("FTS index is maintained automatically by the node store.")


def cmd_doctor(args) -> None:
    from skg.graph import SKG
    from skg.node import NodeStatus
    skg = SKG()
    store = skg._store
    all_nodes = store.list_all()
    active  = [n for n in all_nodes if n.status == NodeStatus.ACTIVE]
    stale   = [n for n in all_nodes if n.status == NodeStatus.STALE]
    cand    = [n for n in all_nodes if n.status == NodeStatus.CANDIDATE]
    errors  = []

    for n in all_nodes:
        if not n.manifest.task_type:
            errors.append(f"  {n.id}: missing task_type in manifest")
        if not n.manifest.header:
            errors.append(f"  {n.id}: missing header in manifest")

    print(f"Total nodes:     {len(all_nodes)}")
    print(f"  Active:        {len(active)}")
    print(f"  Candidate:     {len(cand)}")
    print(f"  Stale:         {len(stale)}")
    if errors:
        print(f"\nSchema errors ({len(errors)}):")
        for e in errors:
            print(e)
    else:
        print("\nAll manifests valid.")
