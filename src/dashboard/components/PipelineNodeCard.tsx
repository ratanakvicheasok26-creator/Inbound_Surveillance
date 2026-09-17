import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";
import { GROUP_ACCENT, NODE_CATALOG } from "../../pipeline/catalog";
import { NODE_PORTS, PORT_COLORS, type PipelineNodeData, type PipelinePort } from "../../pipeline/ir";

export type PipelineFlowNode = Node<PipelineNodeData, "pipeline">;

function Socket({
  port,
  direction,
}: {
  port: PipelinePort;
  direction: "in" | "out";
}) {
  return (
    <div className={`pipe-socket pipe-socket--${direction}`}>
      <Handle
        type={direction === "in" ? "target" : "source"}
        position={direction === "in" ? Position.Left : Position.Right}
        id={port}
        className="pipe-handle"
        style={{ background: PORT_COLORS[port], borderColor: "#050505" }}
        title={port}
      />
      <span style={{ color: PORT_COLORS[port] }}>{port}</span>
    </div>
  );
}

export function PipelineNodeCard({ data, selected }: NodeProps<PipelineFlowNode>) {
  const meta = NODE_CATALOG[data.kind];
  const ports = NODE_PORTS[data.kind];
  const accent = GROUP_ACCENT[meta.group];
  const subtitle =
    data.kind === "zone"
      ? data.zoneName || data.zoneKind || meta.summary
      : data.kind === "camera"
        ? [data.protocol, data.cameraName].filter(Boolean).join(" · ") || meta.summary
        : meta.summary;

  return (
    <div className={`pipe-node${selected ? " is-selected" : ""}`} style={{ borderColor: accent }}>
      <header className="pipe-node__head" style={{ background: `${accent}22`, color: accent }}>
        {meta.label}
      </header>
      <p className="pipe-node__sub">{subtitle}</p>
      <div className="pipe-node__io">
        <div>
          {ports.inputs.map((port) => (
            <Socket key={port} port={port} direction="in" />
          ))}
        </div>
        <div>
          {ports.outputs.map((port) => (
            <Socket key={port} port={port} direction="out" />
          ))}
        </div>
      </div>
    </div>
  );
}
