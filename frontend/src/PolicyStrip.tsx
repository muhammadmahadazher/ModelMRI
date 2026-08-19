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

  const perception = vla?.loaded ?? false;
  const action = policy?.running ?? false;
  // Three states, not two. `null` is "the answer has not arrived", which is
  // NOT "not installed" — the strip used to print "not installed" and a
  // six-gigabyte install instruction during the round trip, on every load,
  // for every user who does have it installed.
  const pending = policy === null && !failed;

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
            {pending
              ? "asking…"
              : failed || !policy
                ? "cannot tell"
                : action
                  ? `contract ${policy.contract} · port ${policy.port}`
                  : policy.reachable
                    ? "running, no policy loaded"
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
              {/* Empty is a fact, not a blank. "these are the same weights"
                  and "nobody recorded which weights" are different claims and
                  a placeholder would collapse them. */}
              {policy.revision ? policy.revision.slice(0, 7) : "revision not recorded"}
              <span className="policy-sep">·</span>
              {policy.device}
              {policy.dtype ? ` ${policy.dtype}` : ""}
              {policy.cameras.length > 0 && (
                <>
                  <span className="policy-sep">·</span>
                  {policy.cameras.length} camera
                  {policy.cameras.length === 1 ? "" : "s"}
                </>
              )}
              {policy.state_width !== null && (
                <>
                  <span className="policy-sep">·</span>
                  state {policy.state_width}
                </>
              )}
              {/* THE ACTION WIDTH, beside the state width it was already
                  showing. It is the number a comparison against a dataset
                  refuses on — "the policy emits 6 action dimensions and this
                  dataset recorded 2" — so a reader who can see it before
                  clicking knows the pairing will not work, and one who cannot
                  finds out after the run is refused. The server has published
                  it all along. */}
              {policy.action_width !== null && policy.action_width !== undefined && (
                <>
                  <span className="policy-sep">·</span>
                  {policy.action_width} action dim
                  {policy.action_width === 1 ? "" : "s"}
                </>
              )}
              {policy.chunk_size !== null && (
                <>
                  <span className="policy-sep">·</span>
                  chunk {policy.chunk_size}
                </>
              )}
            </>
          ) : pending ? (
            "asking the server whether an action expert is installed…"
          ) : failed || !policy ? (
            // A server too old to have the route, or one that did not answer.
            // Saying "not installed" here would be inventing a fact about
            // somebody's machine from a network error.
            "this server did not answer about the action expert, so nothing here knows whether one is installed"
          ) : !policy.installed ? (
            <>
              a second process with its own environment — about {gb.toFixed(0)} GB,
              because lerobot pins torch hard enough that installing it beside
              ModelMRI breaks ModelMRI.{" "}
              {/* The SERVER's sentence, which it has been sending as
                  `install_hint` and nobody read. The command was re-typed here
                  instead, so renaming it would have left this cell confidently
                  prescribing a command that no longer exists — and the copy on
                  screen would still have looked right. */}
              {policy.install_hint ?? <code>modelmri policy install</code>}
            </>
          ) : policy.reachable ? (
            // Up, with no policy in it. Telling somebody to START a sidecar
            // that is answering on a port is the contradiction a review found
            // in this cell: the header already said "running", and the body
            // underneath prescribed the command for the opposite state.
            <>
              A sidecar is answering on port {policy.port} with no policy in
              it. {policy.reason}
            </>
          ) : (
            <>
              <code>modelmri policy start</code> brings it up. {policy.reason}
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
        {pending || failed
          ? perception
            ? "The vision tower is loaded here. Whether anything can say what the policy would DO is still unknown."
            : "Nothing is loaded here yet, and whether an action expert exists on this machine is still unknown."
          : action && perception
          ? "Both halves are available: this can say where the policy looked and what it would do."
          : perception
            ? "This can say where the policy LOOKED. Nothing here can say what it would DO — that needs the action expert."
            : action
              ? "The action expert is up, but no vision tower is loaded here, so there is nothing to show attention over yet."
              : policy?.reachable
                ? "A sidecar is waiting with no policy in it, and no vision tower is loaded here. Neither question can be answered until something is loaded into one of them."
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

      {/* A CPU torch in the sidecar is not wrong, it is FORTY TIMES SLOWER
          than the model running in this process, and nothing else on the page
          would say why a frame takes half a minute. `null` is not `false`:
          nothing has reported a build yet, which is a different situation and
          gets no warning at all. */}
      {action && policy?.accelerated === false && (
        <div className="policy-verdict warn">
          The sidecar holds a CPU build of torch ({policy.torch_version}), so
          the policy runs on the processor while this server runs on the
          accelerator. Answers are still correct and much slower.{" "}
          <code>modelmri policy install --force</code> rebuilds it against this
          machine's card.
        </div>
      )}

      {/* Deterministic is a real property with a real consequence, and it is
          better said here than discovered when a later panel refuses. */}
      {action && policy && !policy.samples && policy.family && (
        <div className="policy-verdict">
          The {policy.family} action head is deterministic — the same frame
          gives the same chunk every time. Anything that needs the policy's own
          sampling spread as a reference has no reference here.
        </div>
      )}
    </div>
  );
}
