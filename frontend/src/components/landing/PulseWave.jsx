"use client";

import { useEffect, useRef } from "react";

// A hand-authored waveform — an actual pulse/ticker line, not a generic
// decorative squiggle. Draws in once on load (stroke animation), then
// drifts slowly with scroll position — one deliberate scroll-linked
// effect, not motion scattered across every element.
const WAVE_PATH =
  "M0,60 L120,60 L150,20 L180,100 L210,40 L240,60 L400,60 L430,10 L460,110 L490,60 L900,60 L930,30 L960,90 L990,60 L1440,60";

export default function PulseWave({ className = "" }) {
  const pathRef = useRef(null);
  const groupRef = useRef(null);

  useEffect(() => {
    const path = pathRef.current;
    const length = path.getTotalLength();
    path.style.strokeDasharray = `${length}`;
    path.style.strokeDashoffset = `${length}`;

    requestAnimationFrame(() => {
      path.style.transition = "stroke-dashoffset 1.8s cubic-bezier(0.16, 1, 0.3, 1)";
      path.style.strokeDashoffset = "0";
    });

    function onScroll() {
      const max = document.body.scrollHeight - window.innerHeight || 1;
      const fraction = Math.min(window.scrollY / max, 1);
      if (groupRef.current) {
        groupRef.current.style.transform = `translateX(${-fraction * 60}px)`;
      }
    }
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <svg viewBox="0 0 1440 120" preserveAspectRatio="none" className={className} aria-hidden="true">
      <g ref={groupRef}>
        <path
          ref={pathRef}
          d={WAVE_PATH}
          fill="none"
          stroke="#C99A4B"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </g>
    </svg>
  );
}