import React from "react";

// Class component: getDerivedStateFromError/componentDidCatch have no hook
// equivalent in React 18. Without this, any uncaught render error (e.g. a
// malformed API response shape) whites out the entire page with only a
// console error — the worst-case outcome for a live portfolio demo.
export class ErrorBoundary extends React.Component {
  state = { hasError: false };

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error, info) {
    console.error("Unhandled UI error:", error, info);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="error-boundary-fallback">
          <h2>Something went wrong.</h2>
          <p className="muted">
            Try reloading the page. If this keeps happening, the API response shape may have
            changed.
          </p>
          <button type="button" onClick={() => window.location.reload()}>
            Reload
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
