"use client";

import { useRouter } from "next/navigation";
import PulseWave from "../components/landing/PulseWave";
import TickerTape from "../components/landing/TickerTape";
import ExplainabilityShowcase from "../components/landing/ExplainabilityShowcase";
import Pipeline from "../components/landing/Pipeline";

export default function LandingPage() {
  const router = useRouter();

  // No real auth flow yet — this is the placeholder redirect. Swap the
  // body of this function for a real login call once auth exists; keep
  // the destination as /dashboard so nothing else needs to change.
  function handleLogin() {
    router.push("/dashboard");
  }

  return (
    <div className="bg-ink text-white">
      <nav className="mx-auto flex max-w-6xl items-center justify-between px-6 py-6">
        <span className="font-display text-xl italic">Pulse</span>
        <button
          onClick={handleLogin}
          className="rounded-full border border-white/20 px-5 py-2 text-sm transition-colors hover:border-white/50"
        >
          Log in
        </button>
      </nav>

      <section className="relative mx-auto max-w-6xl px-6 pb-20 pt-12 sm:pt-20">
        <div className="max-w-2xl">
          <h1 className="font-display text-5xl leading-[1.05] sm:text-6xl">
            Markets move fast.
            <br />
            Your feed should know why.
          </h1>
          <p className="mt-6 max-w-md text-lg leading-relaxed text-white/60">
            Pulse ranks the news and stocks that matter to you, and shows the
            reason behind every recommendation — not a black box, a second
            opinion.
          </p>
          <button
            onClick={handleLogin}
            className="mt-8 rounded-full bg-brass px-7 py-3 text-sm font-medium text-ink transition-transform hover:scale-[1.02]"
          >
            See your Pulse
          </button>
        </div>
        <PulseWave className="pointer-events-none absolute inset-x-0 bottom-0 h-24 w-full opacity-70" />
      </section>

      <TickerTape />

      <section className="mx-auto grid max-w-6xl gap-10 px-6 py-24 sm:grid-cols-2">
        <p className="font-display text-3xl leading-snug text-white/90">
          Most market feeds show everyone the same headlines.
        </p>
        <p className="text-lg leading-relaxed text-white/60">
          Pulse builds a feed around what you actually hold and watch —
          weighing sentiment, sector momentum, and your own behavior — then
          explains its reasoning in plain terms so you can trust or
          challenge it.
        </p>
      </section>

      <section className="mx-auto flex max-w-6xl flex-col items-start gap-10 px-6 py-24 sm:flex-row sm:items-center">
        <div className="max-w-sm">
          <h2 className="font-display text-3xl">Every recommendation shows its work.</h2>
          <p className="mt-4 text-white/60">
            No score without a reason. Pulse surfaces the specific signals —
            sentiment, momentum, your watchlist — behind each item.
          </p>
        </div>
        <ExplainabilityShowcase />
      </section>

      <section className="mx-auto max-w-6xl px-6 py-24">
        <h2 className="max-w-lg font-display text-3xl">Built on two complementary signals.</h2>
        <p className="mt-4 max-w-lg text-white/60">
          A long-term view of what you follow, and a short-term read of what
          you're doing right now.
        </p>
        <div className="mt-12">
          <Pipeline />
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-6 py-24">
        <h2 className="font-display text-4xl">Ready for a feed that explains itself?</h2>
        <button
          onClick={handleLogin}
          className="mt-8 rounded-full bg-brass px-7 py-3 text-sm font-medium text-ink transition-transform hover:scale-[1.02]"
        >
          Get your Pulse
        </button>
      </section>

      <footer className="border-t border-white/10 px-6 py-8 text-sm text-white/40">
        Pulse — built for IMDUAI.
      </footer>
    </div>
  );
}