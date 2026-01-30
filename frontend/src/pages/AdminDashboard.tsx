import React, { useState, useEffect, useCallback } from 'react';
import { 
  Users, DollarSign, Activity, TrendingUp, 
  BarChart3, PieChart, Clock, CheckCircle, XCircle,
  ChevronDown, Search, Filter, RefreshCw, X,
  Camera, Smartphone, Monitor, Crown, Zap
} from 'lucide-react';
import { supabase } from '@/lib/supabase';

interface User {
  id: string;
  email: string;
  display_name: string;
  preferred_language: string;
  created_at: string;
  subscription: {
    plan_type: string;
    api_calls_used: number;
    api_calls_limit: number;
    status: string;
  };
  totalApiCalls: number;
}

interface ApiStats {
  last24h: number;
  last7d: number;
  last30d: number;
  successRate: string;
  cameraBreakdown: {
    webcam: number;
    oak_ai: number;
    lumen: number;
  };
  languageBreakdown: Record<string, number>;
}

interface RevenueStats {
  mrr: number;
  arr: number;
  proCount: number;
  enterpriseCount: number;
  freeCount: number;
  totalPaid: number;
}

interface AdminDashboardProps {
  isOpen: boolean;
  onClose: () => void;
}

const AdminDashboard: React.FC<AdminDashboardProps> = ({ isOpen, onClose }) => {
  const [users, setUsers] = useState<User[]>([]);
  const [apiStats, setApiStats] = useState<ApiStats | null>(null);
  const [revenue, setRevenue] = useState<RevenueStats | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'overview' | 'users' | 'api' | 'revenue'>('overview');
  const [searchQuery, setSearchQuery] = useState('');
  const [planFilter, setPlanFilter] = useState<string>('all');

  const fetchDashboardData = useCallback(async () => {
    setIsLoading(true);
    try {
      const { data, error } = await supabase.functions.invoke('sonzo-admin', {
        body: { action: 'getDashboardStats' }
      });

      if (!error && data) {
        setUsers(data.users || []);
        setApiStats(data.apiStats || null);
        setRevenue(data.revenue || null);
      }
    } catch (err) {
      console.error('Failed to fetch dashboard data:', err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (isOpen) {
      fetchDashboardData();
    }
  }, [isOpen, fetchDashboardData]);

  const handleUpdateSubscription = async (userId: string, planType: string) => {
    try {
      await supabase.functions.invoke('sonzo-admin', {
        body: { action: 'updateSubscription', userId, planType }
      });
      fetchDashboardData();
    } catch (err) {
      console.error('Failed to update subscription:', err);
    }
  };

  const filteredUsers = users.filter(user => {
    const matchesSearch = user.email.toLowerCase().includes(searchQuery.toLowerCase()) ||
      user.display_name?.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesPlan = planFilter === 'all' || user.subscription.plan_type === planFilter;
    return matchesSearch && matchesPlan;
  });

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-card w-full max-w-7xl max-h-[90vh] rounded-2xl border border-border overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-border bg-gradient-to-r from-violet-600/10 to-cyan-600/10">
          <div>
            <h2 className="text-2xl font-bold flex items-center gap-3">
              <Crown className="w-7 h-7 text-yellow-500" />
              Admin Dashboard
            </h2>
            <p className="text-muted-foreground mt-1">Monitor users, subscriptions, and API usage</p>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-muted transition-colors"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex items-center gap-1 p-4 border-b border-border bg-muted/30">
          {(['overview', 'users', 'api', 'revenue'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-2 rounded-lg font-medium capitalize transition-colors ${
                activeTab === tab
                  ? 'bg-primary text-primary-foreground'
                  : 'hover:bg-muted'
              }`}
            >
              {tab}
            </button>
          ))}
          <div className="flex-1" />
          <button
            onClick={fetchDashboardData}
            disabled={isLoading}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-muted hover:bg-muted/80 transition-colors"
          >
            <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto p-6">
          {isLoading ? (
            <div className="flex items-center justify-center h-64">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary" />
            </div>
          ) : (
            <>
              {/* Overview Tab */}
              {activeTab === 'overview' && (
                <div className="space-y-6">
                  {/* Stats Cards */}
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    <div className="bg-gradient-to-br from-violet-500/10 to-violet-500/5 rounded-xl p-5 border border-violet-500/20">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm text-muted-foreground">Total Users</p>
                          <p className="text-3xl font-bold mt-1">{users.length}</p>
                        </div>
                        <div className="p-3 bg-violet-500/20 rounded-xl">
                          <Users className="w-6 h-6 text-violet-500" />
                        </div>
                      </div>
                      <p className="text-xs text-green-500 mt-2 flex items-center gap-1">
                        <TrendingUp className="w-3 h-3" />
                        +12% this month
                      </p>
                    </div>

                    <div className="bg-gradient-to-br from-green-500/10 to-green-500/5 rounded-xl p-5 border border-green-500/20">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm text-muted-foreground">Monthly Revenue</p>
                          <p className="text-3xl font-bold mt-1">${revenue?.mrr?.toLocaleString() || 0}</p>
                        </div>
                        <div className="p-3 bg-green-500/20 rounded-xl">
                          <DollarSign className="w-6 h-6 text-green-500" />
                        </div>
                      </div>
                      <p className="text-xs text-muted-foreground mt-2">
                        ARR: ${revenue?.arr?.toLocaleString() || 0}
                      </p>
                    </div>

                    <div className="bg-gradient-to-br from-cyan-500/10 to-cyan-500/5 rounded-xl p-5 border border-cyan-500/20">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm text-muted-foreground">API Calls (24h)</p>
                          <p className="text-3xl font-bold mt-1">{apiStats?.last24h?.toLocaleString() || 0}</p>
                        </div>
                        <div className="p-3 bg-cyan-500/20 rounded-xl">
                          <Activity className="w-6 h-6 text-cyan-500" />
                        </div>
                      </div>
                      <p className="text-xs text-muted-foreground mt-2">
                        Success rate: {apiStats?.successRate || 100}%
                      </p>
                    </div>

                    <div className="bg-gradient-to-br from-yellow-500/10 to-yellow-500/5 rounded-xl p-5 border border-yellow-500/20">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm text-muted-foreground">Paid Subscribers</p>
                          <p className="text-3xl font-bold mt-1">{revenue?.totalPaid || 0}</p>
                        </div>
                        <div className="p-3 bg-yellow-500/20 rounded-xl">
                          <Crown className="w-6 h-6 text-yellow-500" />
                        </div>
                      </div>
                      <p className="text-xs text-muted-foreground mt-2">
                        Pro: {revenue?.proCount || 0} | Enterprise: {revenue?.enterpriseCount || 0}
                      </p>
                    </div>
                  </div>

                  {/* Charts Row */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Camera Usage */}
                    <div className="bg-muted/30 rounded-xl p-5 border border-border">
                      <h3 className="font-semibold mb-4 flex items-center gap-2">
                        <Camera className="w-5 h-5 text-primary" />
                        Camera Usage (24h)
                      </h3>
                      <div className="space-y-3">
                        {[
                          { label: 'Webcam', value: apiStats?.cameraBreakdown?.webcam || 0, color: 'bg-blue-500', icon: <Monitor className="w-4 h-4" /> },
                          { label: 'OAK AI', value: apiStats?.cameraBreakdown?.oak_ai || 0, color: 'bg-violet-500', icon: <Zap className="w-4 h-4" /> },
                          { label: 'Lumen', value: apiStats?.cameraBreakdown?.lumen || 0, color: 'bg-cyan-500', icon: <Camera className="w-4 h-4" /> }
                        ].map((item) => {
                          const total = (apiStats?.cameraBreakdown?.webcam || 0) + 
                                       (apiStats?.cameraBreakdown?.oak_ai || 0) + 
                                       (apiStats?.cameraBreakdown?.lumen || 0);
                          const percentage = total > 0 ? (item.value / total * 100).toFixed(1) : 0;
                          return (
                            <div key={item.label} className="flex items-center gap-3">
                              <div className={`p-2 rounded-lg ${item.color}/20`}>
                                {item.icon}
                              </div>
                              <div className="flex-1">
                                <div className="flex justify-between text-sm mb-1">
                                  <span>{item.label}</span>
                                  <span className="text-muted-foreground">{item.value} ({percentage}%)</span>
                                </div>
                                <div className="h-2 bg-muted rounded-full overflow-hidden">
                                  <div 
                                    className={`h-full ${item.color} rounded-full transition-all`}
                                    style={{ width: `${percentage}%` }}
                                  />
                                </div>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>

                    {/* Subscription Distribution */}
                    <div className="bg-muted/30 rounded-xl p-5 border border-border">
                      <h3 className="font-semibold mb-4 flex items-center gap-2">
                        <PieChart className="w-5 h-5 text-primary" />
                        Subscription Distribution
                      </h3>
                      <div className="flex items-center justify-center gap-8">
                        <div className="relative w-32 h-32">
                          <svg className="w-full h-full transform -rotate-90">
                            <circle
                              cx="64"
                              cy="64"
                              r="56"
                              fill="none"
                              stroke="currentColor"
                              strokeWidth="16"
                              className="text-muted"
                            />
                            <circle
                              cx="64"
                              cy="64"
                              r="56"
                              fill="none"
                              stroke="currentColor"
                              strokeWidth="16"
                              strokeDasharray={`${(revenue?.proCount || 0) / Math.max(users.length, 1) * 352} 352`}
                              className="text-violet-500"
                            />
                            <circle
                              cx="64"
                              cy="64"
                              r="56"
                              fill="none"
                              stroke="currentColor"
                              strokeWidth="16"
                              strokeDasharray={`${(revenue?.enterpriseCount || 0) / Math.max(users.length, 1) * 352} 352`}
                              strokeDashoffset={`-${(revenue?.proCount || 0) / Math.max(users.length, 1) * 352}`}
                              className="text-yellow-500"
                            />
                          </svg>
                          <div className="absolute inset-0 flex items-center justify-center">
                            <span className="text-2xl font-bold">{users.length}</span>
                          </div>
                        </div>
                        <div className="space-y-2">
                          <div className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded-full bg-muted" />
                            <span className="text-sm">Free: {revenue?.freeCount || 0}</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded-full bg-violet-500" />
                            <span className="text-sm">Pro ($299): {revenue?.proCount || 0}</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded-full bg-yellow-500" />
                            <span className="text-sm">Enterprise ($1999): {revenue?.enterpriseCount || 0}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Users Tab */}
              {activeTab === 'users' && (
                <div className="space-y-4">
                  {/* Filters */}
                  <div className="flex items-center gap-4">
                    <div className="relative flex-1 max-w-md">
                      <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                      <input
                        type="text"
                        placeholder="Search users..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="w-full pl-10 pr-4 py-2 rounded-lg bg-muted border border-border focus:outline-none focus:ring-2 focus:ring-primary"
                      />
                    </div>
                    <select
                      value={planFilter}
                      onChange={(e) => setPlanFilter(e.target.value)}
                      className="px-4 py-2 rounded-lg bg-muted border border-border focus:outline-none focus:ring-2 focus:ring-primary"
                    >
                      <option value="all">All Plans</option>
                      <option value="free">Free</option>
                      <option value="pro">Pro ($299)</option>
                      <option value="enterprise">Enterprise ($1999)</option>
                    </select>
                  </div>

                  {/* Users Table */}
                  <div className="bg-muted/30 rounded-xl border border-border overflow-hidden">
                    <table className="w-full">
                      <thead className="bg-muted/50">
                        <tr>
                          <th className="text-left px-4 py-3 text-sm font-medium">User</th>
                          <th className="text-left px-4 py-3 text-sm font-medium">Plan</th>
                          <th className="text-left px-4 py-3 text-sm font-medium">API Usage</th>
                          <th className="text-left px-4 py-3 text-sm font-medium">Language</th>
                          <th className="text-left px-4 py-3 text-sm font-medium">Joined</th>
                          <th className="text-left px-4 py-3 text-sm font-medium">Actions</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-border">
                        {filteredUsers.map((user) => (
                          <tr key={user.id} className="hover:bg-muted/30 transition-colors">
                            <td className="px-4 py-3">
                              <div>
                                <p className="font-medium">{user.display_name || 'Unnamed'}</p>
                                <p className="text-sm text-muted-foreground">{user.email}</p>
                              </div>
                            </td>
                            <td className="px-4 py-3">
                              <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                                user.subscription.plan_type === 'enterprise' 
                                  ? 'bg-yellow-500/20 text-yellow-500'
                                  : user.subscription.plan_type === 'pro'
                                  ? 'bg-violet-500/20 text-violet-500'
                                  : 'bg-muted text-muted-foreground'
                              }`}>
                                {user.subscription.plan_type.toUpperCase()}
                              </span>
                            </td>
                            <td className="px-4 py-3">
                              <div className="flex items-center gap-2">
                                <div className="w-24 h-2 bg-muted rounded-full overflow-hidden">
                                  <div 
                                    className={`h-full rounded-full ${
                                      user.subscription.api_calls_limit === -1 
                                        ? 'bg-green-500 w-1/4'
                                        : user.subscription.api_calls_used / user.subscription.api_calls_limit > 0.8
                                        ? 'bg-red-500'
                                        : 'bg-primary'
                                    }`}
                                    style={{ 
                                      width: user.subscription.api_calls_limit === -1 
                                        ? '25%' 
                                        : `${Math.min(100, user.subscription.api_calls_used / user.subscription.api_calls_limit * 100)}%` 
                                    }}
                                  />
                                </div>
                                <span className="text-sm text-muted-foreground">
                                  {user.subscription.api_calls_used}/
                                  {user.subscription.api_calls_limit === -1 ? '∞' : user.subscription.api_calls_limit}
                                </span>
                              </div>
                            </td>
                            <td className="px-4 py-3 text-sm">{user.preferred_language || 'ASL'}</td>
                            <td className="px-4 py-3 text-sm text-muted-foreground">
                              {new Date(user.created_at).toLocaleDateString()}
                            </td>
                            <td className="px-4 py-3">
                              <select
                                value={user.subscription.plan_type}
                                onChange={(e) => handleUpdateSubscription(user.id, e.target.value)}
                                className="px-2 py-1 rounded bg-muted border border-border text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                              >
                                <option value="free">Free</option>
                                <option value="pro">Pro ($299)</option>
                                <option value="enterprise">Enterprise ($1999)</option>
                              </select>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    {filteredUsers.length === 0 && (
                      <div className="text-center py-12 text-muted-foreground">
                        No users found
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* API Tab */}
              {activeTab === 'api' && (
                <div className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="bg-muted/30 rounded-xl p-5 border border-border">
                      <div className="flex items-center gap-3 mb-2">
                        <Clock className="w-5 h-5 text-blue-500" />
                        <span className="text-sm text-muted-foreground">Last 24 Hours</span>
                      </div>
                      <p className="text-3xl font-bold">{apiStats?.last24h?.toLocaleString() || 0}</p>
                    </div>
                    <div className="bg-muted/30 rounded-xl p-5 border border-border">
                      <div className="flex items-center gap-3 mb-2">
                        <BarChart3 className="w-5 h-5 text-violet-500" />
                        <span className="text-sm text-muted-foreground">Last 7 Days</span>
                      </div>
                      <p className="text-3xl font-bold">{apiStats?.last7d?.toLocaleString() || 0}</p>
                    </div>
                    <div className="bg-muted/30 rounded-xl p-5 border border-border">
                      <div className="flex items-center gap-3 mb-2">
                        <TrendingUp className="w-5 h-5 text-green-500" />
                        <span className="text-sm text-muted-foreground">Last 30 Days</span>
                      </div>
                      <p className="text-3xl font-bold">{apiStats?.last30d?.toLocaleString() || 0}</p>
                    </div>
                  </div>

                  {/* Language Breakdown */}
                  <div className="bg-muted/30 rounded-xl p-5 border border-border">
                    <h3 className="font-semibold mb-4">Language Usage</h3>
                    <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
                      {Object.entries(apiStats?.languageBreakdown || {}).map(([lang, count]) => (
                        <div key={lang} className="bg-muted/50 rounded-lg p-3 text-center">
                          <p className="text-lg font-bold">{count}</p>
                          <p className="text-sm text-muted-foreground">{lang}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* Revenue Tab */}
              {activeTab === 'revenue' && (
                <div className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    <div className="bg-gradient-to-br from-green-500/10 to-green-500/5 rounded-xl p-5 border border-green-500/20">
                      <p className="text-sm text-muted-foreground">Monthly Recurring Revenue</p>
                      <p className="text-3xl font-bold mt-2">${revenue?.mrr?.toLocaleString() || 0}</p>
                    </div>
                    <div className="bg-gradient-to-br from-blue-500/10 to-blue-500/5 rounded-xl p-5 border border-blue-500/20">
                      <p className="text-sm text-muted-foreground">Annual Recurring Revenue</p>
                      <p className="text-3xl font-bold mt-2">${revenue?.arr?.toLocaleString() || 0}</p>
                    </div>
                    <div className="bg-gradient-to-br from-violet-500/10 to-violet-500/5 rounded-xl p-5 border border-violet-500/20">
                      <p className="text-sm text-muted-foreground">Pro Subscribers</p>
                      <p className="text-3xl font-bold mt-2">{revenue?.proCount || 0}</p>
                      <p className="text-xs text-muted-foreground mt-1">$299/mo each</p>
                    </div>
                    <div className="bg-gradient-to-br from-yellow-500/10 to-yellow-500/5 rounded-xl p-5 border border-yellow-500/20">
                      <p className="text-sm text-muted-foreground">Enterprise Subscribers</p>
                      <p className="text-3xl font-bold mt-2">{revenue?.enterpriseCount || 0}</p>
                      <p className="text-xs text-muted-foreground mt-1">$1,999/mo each</p>
                    </div>
                  </div>

                  {/* Pricing Tiers */}
                  <div className="bg-muted/30 rounded-xl p-6 border border-border">
                    <h3 className="font-semibold mb-4">Pricing Tiers</h3>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div className="bg-card rounded-xl p-5 border border-border">
                        <h4 className="font-semibold">Free</h4>
                        <p className="text-3xl font-bold mt-2">$0<span className="text-sm font-normal text-muted-foreground">/mo</span></p>
                        <ul className="mt-4 space-y-2 text-sm">
                          <li className="flex items-center gap-2">
                            <CheckCircle className="w-4 h-4 text-green-500" />
                            10 API calls/month
                          </li>
                          <li className="flex items-center gap-2">
                            <CheckCircle className="w-4 h-4 text-green-500" />
                            Webcam only
                          </li>
                          <li className="flex items-center gap-2">
                            <XCircle className="w-4 h-4 text-muted-foreground" />
                            No depth sensing
                          </li>
                        </ul>
                      </div>
                      <div className="bg-gradient-to-br from-violet-500/10 to-violet-500/5 rounded-xl p-5 border border-violet-500/30">
                        <div className="flex items-center gap-2">
                          <h4 className="font-semibold">Pro</h4>
                          <span className="px-2 py-0.5 bg-violet-500 text-white text-xs rounded-full">Popular</span>
                        </div>
                        <p className="text-3xl font-bold mt-2">$299<span className="text-sm font-normal text-muted-foreground">/mo</span></p>
                        <ul className="mt-4 space-y-2 text-sm">
                          <li className="flex items-center gap-2">
                            <CheckCircle className="w-4 h-4 text-green-500" />
                            100 API calls/month
                          </li>
                          <li className="flex items-center gap-2">
                            <CheckCircle className="w-4 h-4 text-green-500" />
                            All camera types
                          </li>
                          <li className="flex items-center gap-2">
                            <CheckCircle className="w-4 h-4 text-green-500" />
                            Priority support
                          </li>
                        </ul>
                      </div>
                      <div className="bg-gradient-to-br from-yellow-500/10 to-yellow-500/5 rounded-xl p-5 border border-yellow-500/30">
                        <div className="flex items-center gap-2">
                          <h4 className="font-semibold">Enterprise</h4>
                          <Crown className="w-4 h-4 text-yellow-500" />
                        </div>
                        <p className="text-3xl font-bold mt-2">$1,999<span className="text-sm font-normal text-muted-foreground">/mo</span></p>
                        <ul className="mt-4 space-y-2 text-sm">
                          <li className="flex items-center gap-2">
                            <CheckCircle className="w-4 h-4 text-green-500" />
                            Unlimited API calls
                          </li>
                          <li className="flex items-center gap-2">
                            <CheckCircle className="w-4 h-4 text-green-500" />
                            All camera types
                          </li>
                          <li className="flex items-center gap-2">
                            <CheckCircle className="w-4 h-4 text-green-500" />
                            Dedicated support
                          </li>
                          <li className="flex items-center gap-2">
                            <CheckCircle className="w-4 h-4 text-green-500" />
                            Custom integrations
                          </li>
                        </ul>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default AdminDashboard;
