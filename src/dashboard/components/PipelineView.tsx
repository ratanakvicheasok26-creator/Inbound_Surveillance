import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Background,
  ConnectionMode,
  Controls,
  MiniMap,
  ReactFlow,
  ReactFlowProvider,
  addEdge,
  useEdgesState,
  useNodesState,
  type Connection,
  type Edge,
  type Node,
  type OnSelectionChangeParams,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { PROTOCOLS } from "../account";
import { useAccount } from "../auth";
import { engineBaseUrl } from "../../engine-url";
import type { Json } from "../../lib/database.types";
import { defaultNodeData, GROUP_ACCENT, GROUP_LABELS, GROUP_ORDER, NODE_CATALOG } from "../../pipeline/catalog";
import {
  isPipelinePort,
  PORT_COLORS,
  type PipelineGraph,
  type PipelineNodeData,
  type PipelineNodeKind,
} from "../../pipeline/ir";
import { templateFor } from "../../pipeline/templates";
import { validatePipeline } from "../../pipeline/validate";
import { parseWorkplaceId, workplaceOf } from "../../workplaces";
import { PipelineNodeCard, type PipelineFlowNode } from "./PipelineNodeCard";

const nodeTypes = { pipeline: PipelineNodeCard };

function asGraph(
  workplace: string,
  nodes: Node[],
  edges: Edge[],
): PipelineGraph {
  return {
    workplace_type: parseWorkplaceId(workplace),
    nodes: nodes.map((node) => ({
      id: node.id,
      type: "pipeline",
      position: node.position,
      data: node.data as PipelineNodeData,
    })),
    edges: edges.map((edge) => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      sourceHandle: edge.sourceHandle ?? undefined,
      targetHandle: edge.targetHandle ?? undefined,
    })),
  };
}

function colorEdges(edges: Edge[]): Edge[] {
  return edges.map((edge) => {
    const port = isPipelinePort(edge.sourceHandle) ? edge.sourceHandle : isPipelinePort(edge.targetHandle) ? edge.targetHandle : "frames";
    return {
      ...edge,
      style: { stroke: PORT_COLORS[port], strokeWidth: 2 },
    };
  });
}

function PipelineEditor() {
  const { snapshot, savePipelineGraph } = useAccount();
  const workplaceId = parseWorkplaceId(snapshot?.profile.workplace_type);
  const workplace = workplaceOf(workplaceId);
  const initial = useMemo(() => {
    const stored = snapshot?.pipelineGraph?.graph as PipelineGraph | undefined;
    if (stored?.nodes?.length) return stored;
    return templateFor(workplaceId);
  }, [snapshot?.pipelineGraph?.graph, workplaceId]);
  const [nodes, setNodes, onNodesChange] = useNodesState(initial.nodes as PipelineFlowNode[]);
  const [edges, setEdges, onEdgesChange] = useEdgesState(colorEdges(initial.edges as Edge[]));
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [saveState, setSaveState] = useState<"idle" | "saving" | "saved">("idle");
  const skipAutosave = useRef(true);
  const engine = engineBaseUrl();

  const graph = useMemo(() => asGraph(workplaceId, nodes, edges), [workplaceId, nodes, edges]);
  const errors = useMemo(() => validatePipeline(graph), [graph]);
  const selected = nodes.find((node) => node.id === selectedId) as PipelineFlowNode | undefined;

  const onConnect = useCallback((connection: Connection) => {
    if (!connection.sourceHandle || !connection.targetHandle || connection.sourceHandle !== connection.targetHandle) {
      return;
    }
    const port = isPipelinePort(connection.sourceHandle) ? connection.sourceHandle : "frames";
    setEdges((current) =>
      addEdge(
        {
          ...connection,
          animated: false,
          style: { stroke: PORT_COLORS[port], strokeWidth: 2 },
        },
        current,
      ),
    );
  }, [setEdges]);

  const isValidConnection = useCallback((connection: Connection | Edge) => {
    const src = "sourceHandle" in connection ? connection.sourceHandle : null;
    const dst = "targetHandle" in connection ? connection.targetHandle : null;
    return Boolean(src && dst && src === dst);
  }, []);

  const onSelectionChange = useCallback((params: OnSelectionChangeParams) => {
    setSelectedId(params.nodes[0]?.id ?? null);
  }, []);

  function loadTemplate() {
    const next = templateFor(workplaceId);
    setNodes(next.nodes as PipelineFlowNode[]);
    setEdges(colorEdges(next.edges as Edge[]));
    setNotice(`${workplace.label} template loaded. Save & deploy to apply it to the engine.`);
    setError("");
  }

  function addNode(kind: PipelineNodeKind) {
    const id = `${kind}-${crypto.randomUUID().slice(0, 8)}`;
    setNodes((current) => [
      ...current,
      {
        id,
        type: "pipeline",
        position: { x: 90 + (current.length % 6) * 28, y: 70 + (current.length % 8) * 22 },
        data: defaultNodeData(kind, workplaceId),
      },
    ]);
    setSelectedId(id);
  }

  function patchSelected(partial: Partial<PipelineNodeData>) {
    if (!selectedId) return;
    setNodes((current) =>
      current.map((node) =>
        node.id === selectedId
          ? { ...node, data: { ...node.data, ...partial } }
          : node,
      ),
    );
  }

  function removeSelected() {
    if (!selectedId) return;
    setNodes((current) => current.filter((node) => node.id !== selectedId));
    setEdges((current) => current.filter((edge) => edge.source !== selectedId && edge.target !== selectedId));
    setSelectedId(null);
  }

  const saveRef = useRef(savePipelineGraph);
  saveRef.current = savePipelineGraph;

  useEffect(() => {
    if (skipAutosave.current) {
      skipAutosave.current = false;
      return;
    }
    setSaveState("idle");
    const timer = window.setTimeout(() => {
      setSaveState("saving");
      void saveRef.current(graph as unknown as Json)
        .then(() => setSaveState("saved"))
        .catch((err) => {
          setSaveState("idle");
          setError(err instanceof Error ? err.message : "Could not autosave pipeline.");
        });
    }, 900);
    return () => window.clearTimeout(timer);
  }, [graph]);

  async function saveAndDeploy() {
    if (errors.length) {
      setError(errors.join(" "));
      setNotice("");
      return;
    }
    setBusy(true);
    setError("");
    try {
      await savePipelineGraph(graph as unknown as Json);
      const res = await fetch(`${engine}/api/pipeline`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(graph),
      });
      const data = (await res.json()) as { ok?: boolean; errors?: string[]; error?: string };
      if (!res.ok || data.ok === false) {
        setError((data.errors || [data.error || "Engine rejected the graph"]).join(" "));
        return;
      }
      setSaveState("saved");
      setNotice("Pipeline deployed. Invalid graphs cannot start inference.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not deploy pipeline.");
    } finally {
      setBusy(false);
    }
  }

  const groupedPalette = GROUP_ORDER.map((group) => ({
    group,
    items: Object.values(NODE_CATALOG).filter((item) => item.group === group),
  }));

  const data = (selected?.data || {}) as PipelineNodeData;

  return (
    <section className="panel graph-panel">
      <header className="panel__head">
        <div>
          <h2>Pipeline graph</h2>
          <p>
            This canvas is the source of truth. Typed sockets only connect to the same color.
            Deploy compiles the graph into the edge engine.
          </p>
        </div>
        <div className="actions">
          <span className={`graph-status${errors.length ? " is-bad" : " is-ok"}`}>
            {errors.length ? `${errors.length} issue${errors.length === 1 ? "" : "s"}` : "Valid"}
            {saveState === "saving" ? " · saving" : saveState === "saved" ? " · saved" : ""}
          </span>
          <button className="btn btn--ghost btn--sm" type="button" onClick={loadTemplate}>
            Load {workplace.label} template
          </button>
          <button className="btn btn--primary btn--sm" type="button" disabled={busy} onClick={() => void saveAndDeploy()}>
            {busy ? "Deploying…" : "Queue deploy"}
          </button>
        </div>
      </header>
      {error ? <p className="auth-error">{error}</p> : null}
      {notice ? <p className="auth-notice">{notice}</p> : null}
      {errors.length ? (
        <ul className="graph-errors">
          {errors.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      ) : null}
      <div className="graph-workbench">
        <aside className="graph-palette">
          {groupedPalette.map((section) => (
            <div key={section.group}>
              <h3 style={{ color: GROUP_ACCENT[section.group] }}>{GROUP_LABELS[section.group]}</h3>
              {section.items.map((item) => (
                <button key={item.kind} type="button" onClick={() => addNode(item.kind)}>
                  <strong>{item.label}</strong>
                  <span>{item.summary}</span>
                </button>
              ))}
            </div>
          ))}
        </aside>
        <div className="pipeline-editor">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            isValidConnection={isValidConnection}
            onSelectionChange={onSelectionChange}
            nodeTypes={nodeTypes}
            fitView
            colorMode="dark"
            connectionMode={ConnectionMode.Strict}
            deleteKeyCode={["Backspace", "Delete"]}
            proOptions={{ hideAttribution: true }}
            connectionLineStyle={{ stroke: "#00FF66", strokeWidth: 2 }}
          >
            <Background gap={18} color="#1a1a1a" />
            <MiniMap pannable zoomable nodeColor={(node) => GROUP_ACCENT[NODE_CATALOG[(node.data as PipelineNodeData).kind]?.group] || "#444"} />
            <Controls />
          </ReactFlow>
        </div>
        <aside className="graph-inspector">
          {selected ? (
            <>
              <h3>{NODE_CATALOG[data.kind].label}</h3>
              <p>{NODE_CATALOG[data.kind].summary}</p>
              <label className="field">
                <span>Title</span>
                <input
                  className="nodrag"
                  value={data.label}
                  onChange={(event) => patchSelected({ label: event.target.value })}
                />
              </label>
              {data.kind === "camera" ? (
                <>
                  <label className="field">
                    <span>Camera name</span>
                    <input
                      className="nodrag"
                      value={data.cameraName || ""}
                      onChange={(event) => patchSelected({ cameraName: event.target.value })}
                    />
                  </label>
                  <label className="field">
                    <span>Protocol</span>
                    <select
                      className="nodrag"
                      value={data.protocol || "webcam"}
                      onChange={(event) => patchSelected({ protocol: event.target.value })}
                    >
                      {PROTOCOLS.map((protocol) => (
                        <option key={protocol} value={protocol}>
                          {protocol}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="field">
                    <span>Source URL</span>
                    <input
                      className="nodrag"
                      value={data.sourceUrl || ""}
                      placeholder="optional"
                      onChange={(event) => patchSelected({ sourceUrl: event.target.value })}
                    />
                  </label>
                </>
              ) : null}
              {data.kind === "zone" ? (
                <>
                  <label className="field">
                    <span>Zone name</span>
                    <input
                      className="nodrag"
                      value={data.zoneName || ""}
                      onChange={(event) => patchSelected({ zoneName: event.target.value, label: event.target.value || data.label })}
                    />
                  </label>
                  <label className="field">
                    <span>Kind</span>
                    <select
                      className="nodrag"
                      value={data.zoneKind || workplace.defaultZoneKind}
                      onChange={(event) => patchSelected({ zoneKind: event.target.value })}
                    >
                      {workplace.zoneKinds.map((kind) => (
                        <option key={kind.id} value={kind.id}>
                          {kind.label}
                        </option>
                      ))}
                    </select>
                  </label>
                </>
              ) : null}
              {data.kind === "alertRule" ? (
                <label className="field">
                  <span>Cooldown seconds</span>
                  <input
                    className="nodrag"
                    type="number"
                    min={1}
                    value={data.cooldownSec ?? 30}
                    onChange={(event) => patchSelected({ cooldownSec: Number(event.target.value) || 30 })}
                  />
                </label>
              ) : null}
              <button className="btn btn--danger btn--sm" type="button" onClick={removeSelected}>
                Delete node
              </button>
            </>
          ) : (
            <>
              <h3>Inspector</h3>
              <p>Select a node to edit its sockets and settings. Drag matching colors to wire the pipeline.</p>
            </>
          )}
        </aside>
      </div>
    </section>
  );
}

export function PipelineView() {
  return (
    <ReactFlowProvider>
      <PipelineEditor />
    </ReactFlowProvider>
  );
}
