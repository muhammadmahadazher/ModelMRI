import { useEffect, useState } from "react";
import { DeviceOption, errorText, getDevices } from "./api";

/**
 * Which piece of hardware the next load goes to.
 *
 * WHY THIS EXISTS AT ALL
 *
 * The tool has always detected an accelerator and used it, which is the right
 * default and stays the default. What it could not do was let you disagree.
 * Three real cases have no way to be expressed otherwise: a second card kept
 * free for something else, a card another process is already filling, and a
 * deliberate CPU run to compare a GPU result against. Each of those is
 * somebody knowing something about their machine that no detector can.
 *
 * WHY IT RENDERS NOTHING ON MOST MACHINES
 *
 * One device is not a choice. A laptop with a single GPU gets no control here
 * — the accelerator badge in the topbar already says what it is, and a select
 * with one option is furniture that implies a decision nobody has.
 *
 * WHY "AUTOMATIC" IS AN OPTION RATHER THAN A PRESELECTED DEVICE
 *
 * Preselecting `cuda:0` would look identical and mean something different: it
 * would turn every load into an explicit request, so a machine whose card
 * changed (a second GPU appearing, a driver failing) would keep being sent to
 * a device by name instead of being re-detected. Automatic sends nothing and
 * lets the server choose, which is exactly what happened before this control
 * existed.
 */
export default function DevicePicker({
  value,
  onChange,
  disabled,
}: {
  /** "" is Automatic — nothing is sent and the server chooses. */
  value: string;
  onChange: (device: string) => void;
  disabled?: boolean;
}) {
  const [devices, setDevices] = useState<DeviceOption[] | null>(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    let live = true;
    void getDevices()
      .then((d) => live && setDevices(d.devices))
      .catch((e) => live && setErr(errorText(e)));
    return () => {
      live = false;
    };
  }, []);

  // A failed probe is not a reason to render a broken control. The load still
  // works — it just goes where it always went.
  if (err || !devices || devices.length < 2) return null;

  const gb = (n: number | null) => (n === null ? null : n / 1e9);

  /** What each option says about itself.
   *
   *  Free memory where it is known, because "will this fit" is a question
   *  about what is free rather than what is installed — a card with 24 GB and
   *  400 MB left is the wrong choice and only the second number says so.
   *  Where free is unknown it is OMITTED rather than shown as 0. */
  const describe = (d: DeviceOption) => {
    const free = gb(d.free_bytes);
    const total = gb(d.total_bytes);
    const memory =
      free !== null && total !== null
        ? `${free.toFixed(1)} of ${total.toFixed(1)} GB free`
        : total !== null
          ? `${total.toFixed(1)} GB`
          : "";
    return [d.name, memory, d.dtype].filter(Boolean).join(" · ");
  };

  const chosen = devices.find((d) => d.id === value);
  const auto = devices.find((d) => d.is_default);

  return (
    <label className="device-pick">
      <span className="meta device-pick-label">run on</span>
      <select
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
        aria-label="Which device to load the model onto"
        title={
          chosen
            ? chosen.reason
            : auto
              ? `Automatic — currently ${auto.id}. ${auto.reason}`
              : "Let the tool choose"
        }
      >
        {/* Empty value, deliberately. See the note above on why this is not a
            preselected device id. */}
        <option value="">
          Automatic{auto ? ` (${auto.id})` : ""}
        </option>
        {devices.map((d) => (
          <option key={d.id} value={d.id}>
            {d.id} — {describe(d)}
          </option>
        ))}
      </select>
    </label>
  );
}
