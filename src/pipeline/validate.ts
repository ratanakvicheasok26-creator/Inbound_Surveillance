import { NODE_PORTS, portsCompatible, type PipelineGraph, type PipelineNodeKind, type PipelinePort } from "./ir";

const MONITOR_KINDS = new Set<PipelineNodeKind>(["employeeLabor", "customerVisits"]);

export function validatePipeline(graph: PipelineGraph): string[] {
  const errors: string[] = [];
  const nodes = graph.nodes || [];
  const edges = graph.edges || [];
  const byId = new Map(nodes.map((node) => [node.id, node]));

  if (!nodes.some((node) => node.data.kind === "camera")) {
    errors.push("Graph needs a Camera source.");
  }

  const incoming = new Map<string, number>();
  const incomingPorts = new Map<string, Set<string>>();
  const outgoing = new Map<string, string[]>();
  for (const node of nodes) {
    incoming.set(node.id, 0);
    incomingPorts.set(node.id, new Set());
    outgoing.set(node.id, []);
  }
  for (const edge of edges) {
    if (!byId.has(edge.source) || !byId.has(edge.target)) {
      errors.push(`Edge ${edge.id} points at a missing node.`);
      continue;
    }
    outgoing.get(edge.source)?.push(edge.target);
    incoming.set(edge.target, (incoming.get(edge.target) || 0) + 1);
    const srcKind = byId.get(edge.source)?.data.kind as PipelineNodeKind;
    const dstKind = byId.get(edge.target)?.data.kind as PipelineNodeKind;
    const srcPort = (edge.sourceHandle || NODE_PORTS[srcKind]?.outputs[0]) as PipelinePort;
    const dstPort = (edge.targetHandle || NODE_PORTS[dstKind]?.inputs[0]) as PipelinePort;
    incomingPorts.get(edge.target)?.add(dstPort);
    if (!portsCompatible(srcPort, dstPort)) {
      errors.push(`${srcKind} ${srcPort} cannot connect to ${dstKind} ${dstPort}.`);
    }
  }

  for (const node of nodes) {
    if (!MONITOR_KINDS.has(node.data.kind)) continue;
    const ports = incomingPorts.get(node.id) || new Set();
    if (!ports.has("identity")) {
      errors.push(`${node.data.label || node.data.kind} needs an identity input (Face ID or anonymous re-ID).`);
    }
    if (!ports.has("zones")) {
      errors.push(`${node.data.label || node.data.kind} needs a zone input.`);
    }
  }

  const queue = nodes.filter((node) => (incoming.get(node.id) || 0) === 0).map((node) => node.id);
  let seen = 0;
  const indeg = new Map(incoming);
  while (queue.length) {
    const id = queue.shift() as string;
    seen += 1;
    for (const next of outgoing.get(id) || []) {
      const nextDeg = (indeg.get(next) || 1) - 1;
      indeg.set(next, nextDeg);
      if (nextDeg === 0) queue.push(next);
    }
  }
  if (nodes.length && seen !== nodes.length) {
    errors.push("Graph has a cycle. Disconnect a loop before deploying.");
  }
  return errors;
}
