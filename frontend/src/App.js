import { useEffect, useRef, useState, useCallback } from "react";
import { BrowserRouter, Routes, Route, Navigate, useNavigate, useLocation } from "react-router-dom";
import axios from "axios";
import { Toaster, toast } from "sonner";

// Pages
import Landing from "./pages/Landing";
import Dashboard from "./pages/Dashboard";
import Dictionary from "./pages/Dictionary";
import History from "./pages/History";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
export const API = `${BACKEND_URL}/api`;

// Configure axios defaults
axios.defaults.withCredentials = true;

// Auth Context
import { createContext, useContext } from "react";

const AuthContext = createContext(null);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
};

// REMINDER: DO NOT HARDCODE THE URL, OR ADD ANY FALLBACKS OR REDIRECT URLS, THIS BREAKS THE AUTH
const AuthCallback = () => {
  const navigate = useNavigate();
  const hasProcessed = useRef(false);

  useEffect(() => {
    if (hasProcessed.current) return;
    hasProcessed.current = true;

    const processAuth = async () => {
      const hash = window.location.hash;
      const sessionId = new URLSearchParams(hash.substring(1)).get("session_id");

      if (!sessionId) {
        toast.error("Authentication failed");
        navigate("/");
        return;
      }

      try {
        const response = await axios.post(`${API}/auth/session`, { session_id: sessionId });
        const { user } = response.data;
        
        // Clear the hash and navigate to dashboard with user data
        window.history.replaceState(null, "", window.location.pathname);
        navigate("/dashboard", { state: { user }, replace: true });
        toast.success(`Welcome, ${user.name}!`);
      } catch (error) {
        console.error("Auth error:", error);
        toast.error("Authentication failed. Please try again.");
        navigate("/");
      }
    };

    processAuth();
  }, [navigate]);

  return (
    <div className="min-h-screen bg-background flex items-center justify-center">
      <div className="text-center">
        <div className="w-8 h-8 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
        <p className="text-muted-foreground">Authenticating...</p>
      </div>
    </div>
  );
};

const ProtectedRoute = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState(null);
  const [user, setUser] = useState(null);
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    // If user data was passed from AuthCallback, use it
    if (location.state?.user) {
      setUser(location.state.user);
      setIsAuthenticated(true);
      return;
    }

    const checkAuth = async () => {
      try {
        const response = await axios.get(`${API}/auth/me`);
        setUser(response.data);
        setIsAuthenticated(true);
      } catch (error) {
        setIsAuthenticated(false);
        navigate("/");
      }
    };

    checkAuth();
  }, [navigate, location.state]);

  if (isAuthenticated === null) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/" replace />;
  }

  return (
    <AuthContext.Provider value={{ user, setUser, isAuthenticated }}>
      {children}
    </AuthContext.Provider>
  );
};

function AppRouter() {
  const location = useLocation();
  
  // Check URL fragment for session_id synchronously during render
  if (location.hash?.includes("session_id=")) {
    return <AuthCallback />;
  }

  return (
    <Routes>
      <Route path="/" element={<Landing />} />
      <Route
        path="/dashboard"
        element={
          <ProtectedRoute>
            <Dashboard />
          </ProtectedRoute>
        }
      />
      <Route
        path="/dictionary"
        element={
          <ProtectedRoute>
            <Dictionary />
          </ProtectedRoute>
        }
      />
      <Route
        path="/history"
        element={
          <ProtectedRoute>
            <History />
          </ProtectedRoute>
        }
      />
    </Routes>
  );
}

function App() {
  return (
    <BrowserRouter>
      <Toaster 
        position="top-right" 
        richColors 
        theme="dark"
        toastOptions={{
          style: {
            background: 'hsl(240, 6%, 10%)',
            border: '1px solid hsl(240, 4%, 16%)',
            color: 'hsl(0, 0%, 98%)',
          },
        }}
      />
      <AppRouter />
    </BrowserRouter>
  );
}

export default App;
