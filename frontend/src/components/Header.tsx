import React, { useState } from 'react';
import { 
  Menu, X, Globe, Settings, User, LogOut, 
  ChevronDown, Wifi, WifiOff, Camera, MessageSquare,
  BookOpen, HelpCircle, LogIn, UserPlus,
  BarChart3, Crown, CreditCard, History
} from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';

interface HeaderProps {
  currentLanguage: string;
  onLanguageChange: (lang: string) => void;
  isConnected: boolean;
  onOpenSettings: () => void;
  onOpenAuth: (mode: 'signin' | 'signup') => void;
  onOpenProfile: () => void;
  onOpenAdmin?: () => void;
  onOpenHistory?: () => void;
}

const languages = [
  { code: 'ASL', name: 'American Sign Language', flag: '🇺🇸' },
  { code: 'BSL', name: 'British Sign Language', flag: '🇬🇧' },
  { code: 'ISL', name: 'Indian Sign Language', flag: '🇮🇳' },
  { code: 'LSF', name: 'French Sign Language', flag: '🇫🇷' },
  { code: 'DGS', name: 'German Sign Language', flag: '🇩🇪' },
  { code: 'JSL', name: 'Japanese Sign Language', flag: '🇯🇵' },
  { code: 'Auslan', name: 'Australian Sign Language', flag: '🇦🇺' },
  { code: 'LSM', name: 'Mexican Sign Language', flag: '🇲🇽' },
  { code: 'KSL', name: 'Korean Sign Language', flag: '🇰🇷' },
  { code: 'CSL', name: 'Chinese Sign Language', flag: '🇨🇳' },
  { code: 'RSL', name: 'Russian Sign Language', flag: '🇷🇺' },
  { code: 'LSB', name: 'Brazilian Sign Language', flag: '🇧🇷' },
];

const Header: React.FC<HeaderProps> = ({ 
  currentLanguage, 
  onLanguageChange, 
  isConnected,
  onOpenSettings,
  onOpenAuth,
  onOpenProfile,
  onOpenAdmin,
  onOpenHistory
}) => {
  const { user, isAuthenticated, signOut } = useAuth();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [languageDropdownOpen, setLanguageDropdownOpen] = useState(false);
  const [userMenuOpen, setUserMenuOpen] = useState(false);

  const currentLang = languages.find(l => l.code === currentLanguage) || languages[0];
  const isAdmin = isAuthenticated;

  const navItems = [
    { icon: Camera, label: 'Recognition', href: '#recognition' },
    { icon: MessageSquare, label: 'Conversation', href: '#conversation' },
    { icon: BookOpen, label: 'Learn', href: '#learn' },
    { icon: CreditCard, label: 'Pricing', href: '#pricing' },
    { icon: HelpCircle, label: 'FAQ', href: '#faq' },
  ];

  const handleSignOut = async () => {
    await signOut();
    setUserMenuOpen(false);
  };

  return (
    <header className="fixed top-0 left-0 right-0 z-50 bg-background/80 backdrop-blur-xl border-b border-border">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center shadow-lg shadow-violet-500/30">
              <svg viewBox="0 0 24 24" className="w-6 h-6 text-white" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M7 11V7a5 5 0 0 1 10 0v4" strokeLinecap="round" />
                <path d="M12 11v6" strokeLinecap="round" />
                <path d="M8 15h8" strokeLinecap="round" />
                <circle cx="12" cy="18" r="1" fill="currentColor" />
                <path d="M5 11h2v8a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2v-8h2" strokeLinecap="round" />
              </svg>
            </div>
            <div>
              <h1 className="text-xl font-bold gradient-text">SonZo AI</h1>
              <p className="text-[10px] text-muted-foreground -mt-1">Sign Language Recognition</p>
            </div>
          </div>

          {/* Desktop Navigation */}
          <nav className="hidden lg:flex items-center gap-1">
            {navItems.map((item) => (
              <a
                key={item.label}
                href={item.href}
                className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
              >
                <item.icon className="w-4 h-4" />
                {item.label}
              </a>
            ))}
          </nav>

          {/* Right Side Actions */}
          <div className="flex items-center gap-3">
            {/* Connection Status */}
            <div className={`hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium ${
              isConnected 
                ? 'bg-green-500/10 text-green-600 dark:text-green-400' 
                : 'bg-red-500/10 text-red-600 dark:text-red-400'
            }`}>
              {isConnected ? (
                <>
                  <Wifi className="w-3.5 h-3.5" />
                  <span>Connected</span>
                </>
              ) : (
                <>
                  <WifiOff className="w-3.5 h-3.5" />
                  <span>Disconnected</span>
                </>
              )}
            </div>

            {/* Language Selector */}
            <div className="relative">
              <button
                onClick={() => setLanguageDropdownOpen(!languageDropdownOpen)}
                className="flex items-center gap-2 px-3 py-2 rounded-lg bg-muted hover:bg-muted/80 transition-colors"
              >
                <Globe className="w-4 h-4 text-primary" />
                <span className="hidden sm:inline text-sm font-medium">{currentLang.code}</span>
                <ChevronDown className={`w-4 h-4 transition-transform ${languageDropdownOpen ? 'rotate-180' : ''}`} />
              </button>

              {languageDropdownOpen && (
                <>
                  <div 
                    className="fixed inset-0 z-40" 
                    onClick={() => setLanguageDropdownOpen(false)} 
                  />
                  <div className="absolute right-0 mt-2 w-72 bg-card border border-border rounded-xl shadow-xl z-50 overflow-hidden">
                    <div className="p-3 border-b border-border">
                      <h3 className="font-semibold text-sm">Select Sign Language</h3>
                      <p className="text-xs text-muted-foreground mt-0.5">12 languages available globally</p>
                    </div>
                    <div className="max-h-80 overflow-y-auto p-2">
                      {languages.map((lang) => (
                        <button
                          key={lang.code}
                          onClick={() => {
                            onLanguageChange(lang.code);
                            setLanguageDropdownOpen(false);
                          }}
                          className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-left transition-colors ${
                            currentLanguage === lang.code 
                              ? 'bg-primary/10 text-primary' 
                              : 'hover:bg-muted'
                          }`}
                        >
                          <span className="text-xl">{lang.flag}</span>
                          <div className="flex-1">
                            <p className="text-sm font-medium">{lang.code}</p>
                            <p className="text-xs text-muted-foreground">{lang.name}</p>
                          </div>
                          {currentLanguage === lang.code && (
                            <div className="w-2 h-2 rounded-full bg-primary" />
                          )}
                        </button>
                      ))}
                    </div>
                  </div>
                </>
              )}
            </div>

            {/* Settings Button */}
            <button
              onClick={onOpenSettings}
              className="p-2 rounded-lg hover:bg-muted transition-colors"
              aria-label="Settings"
            >
              <Settings className="w-5 h-5" />
            </button>

            {/* User Menu / Auth Buttons */}
            {isAuthenticated && user ? (
              <div className="relative">
                <button
                  onClick={() => setUserMenuOpen(!userMenuOpen)}
                  className="flex items-center gap-2 p-1 rounded-lg hover:bg-muted transition-colors"
                >
                  <div className="w-8 h-8 rounded-full bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center overflow-hidden">
                    {user.avatarUrl ? (
                      <img src={user.avatarUrl} alt={user.displayName} className="w-full h-full object-cover" />
                    ) : (
                      <span className="text-sm font-bold text-white">
                        {user.displayName?.charAt(0).toUpperCase() || user.email.charAt(0).toUpperCase()}
                      </span>
                    )}
                  </div>
                </button>

                {userMenuOpen && (
                  <>
                    <div 
                      className="fixed inset-0 z-40" 
                      onClick={() => setUserMenuOpen(false)} 
                    />
                    <div className="absolute right-0 mt-2 w-64 bg-card border border-border rounded-xl shadow-xl z-50 overflow-hidden">
                      <div className="p-4 border-b border-border">
                        <div className="flex items-center gap-3">
                          <div className="w-12 h-12 rounded-full bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center overflow-hidden">
                            {user.avatarUrl ? (
                              <img src={user.avatarUrl} alt={user.displayName} className="w-full h-full object-cover" />
                            ) : (
                              <span className="text-lg font-bold text-white">
                                {user.displayName?.charAt(0).toUpperCase() || user.email.charAt(0).toUpperCase()}
                              </span>
                            )}
                          </div>
                          <div className="flex-1 min-w-0">
                            <p className="font-semibold truncate">{user.displayName}</p>
                            <p className="text-xs text-muted-foreground truncate">{user.email}</p>
                          </div>
                        </div>
                      </div>
                      <div className="p-2">
                        <button 
                          onClick={() => {
                            onOpenProfile();
                            setUserMenuOpen(false);
                          }}
                          className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-left hover:bg-muted transition-colors"
                        >
                          <User className="w-4 h-4" />
                          <span className="text-sm">My Profile</span>
                        </button>
                        <button 
                          onClick={() => {
                            onOpenProfile();
                            setUserMenuOpen(false);
                          }}
                          className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-left hover:bg-muted transition-colors"
                        >
                          <BarChart3 className="w-4 h-4" />
                          <span className="text-sm">My Progress</span>
                        </button>
                        {onOpenHistory && (
                          <button 
                            onClick={() => {
                              onOpenHistory();
                              setUserMenuOpen(false);
                            }}
                            className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-left hover:bg-muted transition-colors"
                          >
                            <History className="w-4 h-4" />
                            <span className="text-sm">Conversation History</span>
                          </button>
                        )}
                        <button 
                          onClick={() => {
                            onOpenSettings();
                            setUserMenuOpen(false);
                          }}
                          className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-left hover:bg-muted transition-colors"
                        >
                          <Settings className="w-4 h-4" />
                          <span className="text-sm">Settings</span>
                        </button>
                        {isAdmin && onOpenAdmin && (
                          <>
                            <hr className="my-2 border-border" />
                            <button 
                              onClick={() => {
                                onOpenAdmin();
                                setUserMenuOpen(false);
                              }}
                              className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-left text-yellow-500 hover:bg-yellow-500/10 transition-colors"
                            >
                              <Crown className="w-4 h-4" />
                              <span className="text-sm">Admin Dashboard</span>
                            </button>
                          </>
                        )}
                        <hr className="my-2 border-border" />
                        <button 
                          onClick={handleSignOut}
                          className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-left text-red-500 hover:bg-red-500/10 transition-colors"
                        >
                          <LogOut className="w-4 h-4" />
                          <span className="text-sm">Sign Out</span>
                        </button>
                      </div>
                    </div>
                  </>
                )}
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <button
                  onClick={() => onOpenAuth('signin')}
                  className="hidden sm:flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium hover:bg-muted transition-colors"
                >
                  <LogIn className="w-4 h-4" />
                  Sign In
                </button>
                <button
                  onClick={() => onOpenAuth('signup')}
                  className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg text-sm font-medium hover:bg-primary/90 transition-colors"
                >
                  <UserPlus className="w-4 h-4" />
                  <span className="hidden sm:inline">Sign Up</span>
                </button>
              </div>
            )}

            {/* Mobile Menu Button */}
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="lg:hidden p-2 rounded-lg hover:bg-muted transition-colors"
            >
              {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Menu */}
      {mobileMenuOpen && (
        <div className="lg:hidden border-t border-border bg-background">
          <nav className="p-4 space-y-1">
            {navItems.map((item) => (
              <a
                key={item.label}
                href={item.href}
                onClick={() => setMobileMenuOpen(false)}
                className="flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
              >
                <item.icon className="w-5 h-5" />
                {item.label}
              </a>
            ))}
            
            {/* Mobile History Button for authenticated users */}
            {isAuthenticated && onOpenHistory && (
              <button
                onClick={() => {
                  onOpenHistory();
                  setMobileMenuOpen(false);
                }}
                className="w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
              >
                <History className="w-5 h-5" />
                Conversation History
              </button>
            )}
            
            {/* Mobile Auth Buttons */}
            {!isAuthenticated && (
              <div className="pt-4 border-t border-border mt-4 space-y-2">
                <button
                  onClick={() => {
                    onOpenAuth('signin');
                    setMobileMenuOpen(false);
                  }}
                  className="w-full flex items-center justify-center gap-2 px-4 py-3 rounded-lg text-sm font-medium hover:bg-muted transition-colors"
                >
                  <LogIn className="w-5 h-5" />
                  Sign In
                </button>
                <button
                  onClick={() => {
                    onOpenAuth('signup');
                    setMobileMenuOpen(false);
                  }}
                  className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-primary text-primary-foreground rounded-lg text-sm font-medium hover:bg-primary/90 transition-colors"
                >
                  <UserPlus className="w-5 h-5" />
                  Sign Up
                </button>
              </div>
            )}
          </nav>
        </div>
      )}
    </header>
  );
};

export default Header;
