import { SubmitPanel } from "./components/SubmitPanel";
import "./App.css";

function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>Lexory</h1>
        <p>Submit English text, get lessons and exercises.</p>
      </header>
      <main>
        <SubmitPanel />
      </main>
    </div>
  );
}

export default App;
