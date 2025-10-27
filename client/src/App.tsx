import "./App.css";
import Dashboard from "./components/Dashboard";
import Navbar from "./components/Navbar";
function App() {
  return (
    <div className="w-full p-4">
      <Navbar></Navbar>
      <Dashboard></Dashboard>
    </div>
  );
}

export default App;
