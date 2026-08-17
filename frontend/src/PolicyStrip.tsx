import { useEffect, useState } from "react";
import { getPolicy, PolicyStatus, VLAStatus } from "./api";

/** The two halves of a robot policy, shown as two halves.
 *
 *  A VLA is a vision tower and an action expert, and on almost every machine
 *  exactly one of them is available: the tower loads in this process, the
 *  expert needs lerobot, and lerobot's pins cannot share an environment with
 *  ModelMRI's. So it lives in a second process with its own venv.
 *
 *  That is why this is two cells rather than one status light. A single
 *  "VLA: loaded" would have to pick one of the two to speak for, and whichever
 *  it picked would be a claim about the other that nobody checked. The panel
 *  underneath can say where a policy LOOKED; it can only say what the policy
 *  would DO when the right-hand cell is green.
 *
 *  Nothing here starts anything. Installing the action expert is six gigabytes
 *  and a command; this says so and names the command rather than offering a
 *  button that spends that much on one click.
 */
export default function PolicyStrip({ vla }: { vla: VLAStatus | null }) {
  const [policy, setPolicy] = useState<PolicyStatus | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let live = true;
    void getPolicy()
      .then((p) => live && setPolicy(p))
      .catch(() => live && setFailed(true));
    return () => {
      live = false;
    };
  }, []);

  // A server too old to have the route is not the same as a machine with no
  // sidecar, and inventing "not installed" for it would be answering a
  // question nobody asked. Show nothing rather than something wrong.
  if (failed) return null;

  const perception = vla?.loaded ?? false;
  const action = policy?.running ?? false;

  const gb = policy ? policy.venv_disk_bytes / 1e9 : 0;

  return (
    <div className="policy-strip">
      <div className={`policy-half${perception ? " on" : ""}`}>
        <div className="policy-head">
          <span className={`policy-led${perception ? " on" : ""}`} />
          <span className="policy-label">PERCEPTION</span>
          <span className="policy-state">
            {perception ? "loaded here" : "not loaded"}
          </span>
        </div>
        <div className="policy-body">
          {perception ? (
            <>
              {vla?.repo}
              <span className="policy-sep">·</span>
              {vla?.n_layers}L × {vla?.n_heads}H
              <span className="policy-sep">·</span>
              {vla?.grid.join("×")} patches
            </>
          ) : (
            "the vision tower runs in this process — load one to see where a policy looked"
          )}
        </div>
      </div>

      <div className={`policy-half${action ? " on" : ""}`}>
        <div className="policy-head">
          <span className={`policy-led${action ? " on" : ""}`} />
          <span className="policy-label">ACTION</span>
          <span className="policy-state">
            {!policy
              ? "asking…"
              : action
                ? `contract ${policy.contract} · port ${policy.port}`
                : policy.installed
                  ? "installed, not running"
                  : "not installed"}
          </span>
        </div>
        <div className="policy-body">
          {action && policy ? (
            <>
              {policy.policy_repo || "a policy"}
              <span className="policy-sep">·</span>
              {policy.revision ? policy.revision.slice(0, 7) : "revision not recorded"}
              <span className="policy-sep">·</span>
              {policy.device}
              {policy.dtype ? ` ${policy.dtype}` : ""}
            </>
          ) : policy && !policy.installed ? (
            <>
              a second process with its own environment — about {gb.toFixed(0)} GB,
              because lerobot pins torch hard enough that installing it beside
              ModelMRI breaks ModelMRI. <code>modelmri policy install</code>
            </>
          ) : (
            <>
              <code>modelmri policy start</code> brings it up.{" "}
              {policy?.reason ?? ""}
            </>
          )}
        </div>
      </div>

      {/* One sentence about what the two states together permit. Written from
          the measured pair rather than from either half, because the useful
          fact is the combination: a loaded tower with no action expert is a
          real, common and perfectly valid configuration that answers exactly
          one of the two questions somebody came here with. */}
      <div className="policy-verdict">
        {action && perception
          ? "Both halves are available: this can say where the policy looked and what it would do."
          : perception
            ? "This can say where the policy LOOKED. Nothing here can say what it would DO — that needs the action expert."
            : action
              ? "The action expert is up, but no vision tower is loaded here, so there is nothing to show attention over yet."
              : "Neither half is loaded, so nothing on this page is measuring a policy yet."}
      </div>

      {/* Units, and only when there is a policy to have units. Empty
          normalisation is not "identity" — it means the policy never published
          the statistics its actions are scaled against, and drawing them over
          a dataset's recorded actions would be overlaying two different
          units on one axis. */}
      {action && policy && Object.keys(policy.normalisation).length === 0 && (
        <div className="policy-verdict warn">
          This policy does not publish its action statistics, so its actions
          cannot be overlaid on a dataset's recorded ones — the two would be in
          different units with nothing to say so.
        </div>
      )}
    </div>
  );
}
