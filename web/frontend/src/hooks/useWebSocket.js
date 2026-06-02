/**
 * useWebSocket.js — WebSocket Connection Lifecycle Hook
 *
 * This is the ONLY file in the frontend that opens or closes a WebSocket
 * connection. No other file may instantiate or manage a WebSocket directly.
 *
 * Exposes:
 *   - payload      {object|null}   Parsed JSON from the last received message.
 *                                  Shape matches ws_broadcaster.py output:
 *                                  { type, metrics, prediction, active_path,
 *                                    switching_events }
 *   - isConnected  {boolean}       True when the socket is in OPEN state.
 *   - lastUpdated  {Date|null}     Timestamp of the last successful message.
 *
 * Behaviour:
 *   - Connects to VITE_WS_URL (from .env) on first mount.
 *   - Reconnects automatically on close/error with exponential backoff
 *     (base 1 s, max 30 s, jitter ±20 %).
 *   - Cleans up (closes socket, clears timers) on component unmount.
 *   - Ignores non-"update" message types (e.g. the "connected" handshake).
 *   - Contains NO business logic — only raw connection + JSON parse.
 */

import { useState, useEffect, useRef, useCallback } from 'react'

const WS_URL = import.meta.env.VITE_WS_URL ?? 'ws://127.0.0.1:8000/ws'

// Backoff config
const BACKOFF_BASE_MS   = 1_000   // initial reconnect delay
const BACKOFF_MAX_MS    = 30_000  // ceiling
const BACKOFF_JITTER    = 0.2     // ± 20 % random jitter

function nextDelay(attempt) {
  const exp = Math.min(BACKOFF_BASE_MS * 2 ** attempt, BACKOFF_MAX_MS)
  const jitter = exp * BACKOFF_JITTER * (Math.random() * 2 - 1)
  return Math.round(exp + jitter)
}

export function useWebSocket() {
  const [payload, setPayload]         = useState(null)
  const [isConnected, setIsConnected] = useState(false)
  const [lastUpdated, setLastUpdated] = useState(null)

  // Refs so callbacks always see current values without re-running the effect.
  const wsRef      = useRef(null)
  const attemptRef = useRef(0)
  const timerRef   = useRef(null)
  const unmounted  = useRef(false)

  const connect = useCallback(() => {
    if (unmounted.current) return

    // Close any existing stale socket before opening a new one.
    if (wsRef.current) {
      wsRef.current.onclose = null  // suppress the reconnect-on-close handler
      wsRef.current.onerror = null
      wsRef.current.close()
      wsRef.current = null
    }

    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    ws.onopen = () => {
      if (unmounted.current) { ws.close(); return }
      attemptRef.current = 0
      setIsConnected(true)
    }

    ws.onmessage = (event) => {
      if (unmounted.current) return
      try {
        const data = JSON.parse(event.data)
        // Skip the initial handshake message — only store "update" payloads.
        if (data?.type === 'update') {
          setPayload(data)
          setLastUpdated(new Date())
        }
      } catch {
        // Malformed JSON — ignore silently; do not crash the loop.
      }
    }

    ws.onclose = () => {
      if (unmounted.current) return
      setIsConnected(false)
      wsRef.current = null
      // Schedule reconnect with backoff.
      const delay = nextDelay(attemptRef.current)
      attemptRef.current += 1
      timerRef.current = setTimeout(connect, delay)
    }

    ws.onerror = () => {
      // onclose fires right after onerror — let it handle reconnect.
      ws.close()
    }
  }, [])   // stable reference; WS_URL and helpers are module-level constants

  useEffect(() => {
    unmounted.current = false
    connect()

    return () => {
      unmounted.current = true
      clearTimeout(timerRef.current)
      if (wsRef.current) {
        wsRef.current.onclose = null
        wsRef.current.onerror = null
        wsRef.current.close()
        wsRef.current = null
      }
    }
  }, [connect])

  return { payload, isConnected, lastUpdated }
}
