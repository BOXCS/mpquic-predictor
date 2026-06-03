/**
 * useMetrics.js — Metrics Data Layer Hook
 *
 * The ONLY hook components are allowed to import for live network data.
 * Consumes useWS() internally and exposes structured, chart-ready state.
 * Contains no JSX and no WebSocket management — transformation logic only.
 *
 * Exposes:
 *   metrics          {object}  Per-path rolling history.
 *                              Shape: { 1: MetricPoint[], 2: MetricPoint[] }
 *                              MetricPoint: { rtt_ms, goodput_bps, loss_pct, timestamp }
 *                              Capped to WINDOW_SIZE entries per path (FIFO).
 *
 *   latestPerPath    {object}  Most recent single reading per path.
 *                              Shape: { 1: MetricPoint|null, 2: MetricPoint|null }
 *                              Used for stat cards that show a single live number.
 *
 *   prediction       {object|null}
 *                              { predicted_path, confidence, degradation_detected,
 *                                timestamp } or null when no prediction yet.
 *
 *   activePath       {object|null}
 *                              { active_path, path_label, health[] } or null.
 *
 *   switchingEvents  {Array}   Cumulative log of all switching events received
 *                              during this browser session (newest first).
 *
 *   isConnected      {boolean} Forwarded from useWS.
 *   lastUpdated      {Date|null} Forwarded from useWS.
 *
 * Invariants:
 *   - Never throws on null/malformed payload — every access is guarded.
 *   - metrics rolling window is capped at WINDOW_SIZE (60 points ≈ 1 minute
 *     at 1 s push interval).
 *   - switchingEvents are deduplicated by event id to avoid duplicates when
 *     the backend re-sends previously seen events.
 */

import { useState, useEffect, useRef } from 'react'
import { useWebSocket } from '../hooks/useWebSocket'

/** Maximum metric data points retained per path. */
const WINDOW_SIZE = 60

/** Safely extract an array from a payload field, returning [] on bad input. */
function safeArray(val) {
  return Array.isArray(val) ? val : []
}

export function useMetrics() {
  const { payload, isConnected, lastUpdated } = useWebSocket()

  // Rolling per-path history: { [path_id]: MetricPoint[] }
  const [metrics, setMetrics] = useState({ 1: [], 2: [] })

  // Latest single reading per path: { [path_id]: MetricPoint | null }
  const [latestPerPath, setLatestPerPath] = useState({ 1: null, 2: null })

  // Latest LSTM prediction
  const [prediction, setPrediction] = useState(null)

  // Active path + per-path health state
  const [activePath, setActivePath] = useState(null)

  // Cumulative switching events (deduplicated by id, newest first)
  const [switchingEvents, setSwitchingEvents] = useState([])

  // Track seen event ids across renders to deduplicate.
  const seenEventIds = useRef(new Set())

  useEffect(() => {
    // Guard: ignore null, non-update types, or structurally incomplete payloads.
    if (!payload || payload.type !== 'update') return

    // ── 1. Update rolling metrics per path ──────────────────────────────────
    const incomingMetrics = safeArray(payload.metrics)
    if (incomingMetrics.length > 0) {
      setMetrics((prev) => {
        const next = { ...prev }
        for (const point of incomingMetrics) {
          const pid = point?.path_id
          if (pid == null) continue
          const existing = Array.isArray(next[pid]) ? next[pid] : []
          const updated = [
            ...existing,
            {
              rtt_ms: point.rtt_ms ?? null,
              goodput_bps: point.goodput_bps ?? null,
              loss_pct: point.loss_pct ?? null,
              timestamp: point.timestamp ?? null,
            },
          ]
          // Enforce FIFO cap
          next[pid] = updated.length > WINDOW_SIZE
            ? updated.slice(updated.length - WINDOW_SIZE)
            : updated
        }
        return next
      })

      // Derive latest-per-path from the incoming slice (not rolling history).
      setLatestPerPath((prev) => {
        const next = { ...prev }
        for (const point of incomingMetrics) {
          const pid = point?.path_id
          if (pid == null) continue
          next[pid] = {
            rtt_ms: point.rtt_ms ?? null,
            goodput_bps: point.goodput_bps ?? null,
            loss_pct: point.loss_pct ?? null,
            timestamp: point.timestamp ?? null,
          }
        }
        return next
      })
    }

    // ── 2. Update prediction ─────────────────────────────────────────────────
    if (payload.prediction != null) {
      setPrediction({
        predicted_path: payload.prediction.predicted_path ?? null,
        confidence: payload.prediction.confidence ?? null,
        degradation_detected: payload.prediction.degradation_detected ?? false,
        timestamp: payload.prediction.timestamp ?? null,
      })
    }

    // ── 3. Update active path ────────────────────────────────────────────────
    if (payload.active_path != null) {
      setActivePath(payload.active_path)
    }

    // ── 4. Accumulate new switching events (deduplicated) ───────────────────
    const incoming = safeArray(payload.switching_events)
    const novel = incoming.filter(
      (e) => e?.id != null && !seenEventIds.current.has(e.id)
    )
    if (novel.length > 0) {
      for (const e of novel) seenEventIds.current.add(e.id)
      // Prepend newest events; slice to prevent unbounded growth in long sessions.
      setSwitchingEvents((prev) => [...novel, ...prev].slice(0, 200))
    }
  }, [payload])

  return {
    metrics,
    latestPerPath,
    prediction,
    activePath,
    switchingEvents,
    isConnected,
    lastUpdated,
  }
}
