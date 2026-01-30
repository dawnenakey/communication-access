import React, { useState } from 'react';
import { 
  X, Camera, Cpu, User, Monitor, 
  Volume2, Bell, Moon, Sun, Sliders,
  Save, RotateCcw
} from 'lucide-react';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  settings: AppSettings;
  onSaveSettings: (settings: AppSettings) => void;
}

export interface AppSettings {
  camera: {
    resolution: string;
    fps: number;
    depthEnabled: boolean;
    landmarksVisible: boolean;
  };
  model: {
    confidenceThreshold: number;
    sentenceBufferSize: number;
    autoCorrect: boolean;
  };
  avatar: {
    signSpeed: number;
    showSubtitles: boolean;
    voiceEnabled: boolean;
  };
  display: {
    theme: 'light' | 'dark' | 'system';
    notifications: boolean;
    soundEffects: boolean;
  };
}

const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  settings,
  onSaveSettings
}) => {
  const [localSettings, setLocalSettings] = useState<AppSettings>(settings);
  const [activeTab, setActiveTab] = useState<'camera' | 'model' | 'avatar' | 'display'>('camera');

  if (!isOpen) return null;

  const handleSave = () => {
    onSaveSettings(localSettings);
    onClose();
  };

  const handleReset = () => {
    setLocalSettings({
      camera: {
        resolution: '1280x720',
        fps: 30,
        depthEnabled: true,
        landmarksVisible: true
      },
      model: {
        confidenceThreshold: 0.7,
        sentenceBufferSize: 90,
        autoCorrect: true
      },
      avatar: {
        signSpeed: 1,
        showSubtitles: true,
        voiceEnabled: false
      },
      display: {
        theme: 'system',
        notifications: true,
        soundEffects: true
      }
    });
  };

  const tabs = [
    { id: 'camera' as const, label: 'Camera', icon: Camera },
    { id: 'model' as const, label: 'Model', icon: Cpu },
    { id: 'avatar' as const, label: 'Avatar', icon: User },
    { id: 'display' as const, label: 'Display', icon: Monitor }
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative w-full max-w-2xl bg-card rounded-2xl border border-border shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-border">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center">
              <Sliders className="w-5 h-5 text-primary" />
            </div>
            <div>
              <h2 className="text-xl font-bold">Settings</h2>
              <p className="text-sm text-muted-foreground">Configure SonZo AI</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-muted transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-border">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 text-sm font-medium transition-colors ${
                activeTab === tab.id
                  ? 'text-primary border-b-2 border-primary bg-primary/5'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              <tab.icon className="w-4 h-4" />
              {tab.label}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="p-6 max-h-[60vh] overflow-y-auto">
          {/* Camera Settings */}
          {activeTab === 'camera' && (
            <div className="space-y-6">
              <div>
                <label className="block text-sm font-medium mb-2">Resolution</label>
                <select
                  value={localSettings.camera.resolution}
                  onChange={(e) => setLocalSettings({
                    ...localSettings,
                    camera: { ...localSettings.camera, resolution: e.target.value }
                  })}
                  className="w-full px-4 py-2 rounded-lg bg-background border border-border focus:border-primary outline-none"
                >
                  <option value="640x480">640x480 (SD)</option>
                  <option value="1280x720">1280x720 (HD)</option>
                  <option value="1920x1080">1920x1080 (Full HD)</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Frame Rate: {localSettings.camera.fps} FPS</label>
                <input
                  type="range"
                  min="15"
                  max="60"
                  value={localSettings.camera.fps}
                  onChange={(e) => setLocalSettings({
                    ...localSettings,
                    camera: { ...localSettings.camera, fps: parseInt(e.target.value) }
                  })}
                  className="w-full accent-primary"
                />
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">Depth Sensing</p>
                  <p className="text-sm text-muted-foreground">Enable OAK camera depth data</p>
                </div>
                <button
                  onClick={() => setLocalSettings({
                    ...localSettings,
                    camera: { ...localSettings.camera, depthEnabled: !localSettings.camera.depthEnabled }
                  })}
                  className={`w-12 h-6 rounded-full transition-colors ${
                    localSettings.camera.depthEnabled ? 'bg-primary' : 'bg-muted'
                  }`}
                >
                  <div className={`w-5 h-5 rounded-full bg-white shadow transition-transform ${
                    localSettings.camera.depthEnabled ? 'translate-x-6' : 'translate-x-0.5'
                  }`} />
                </button>
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">Show Landmarks</p>
                  <p className="text-sm text-muted-foreground">Display hand tracking points</p>
                </div>
                <button
                  onClick={() => setLocalSettings({
                    ...localSettings,
                    camera: { ...localSettings.camera, landmarksVisible: !localSettings.camera.landmarksVisible }
                  })}
                  className={`w-12 h-6 rounded-full transition-colors ${
                    localSettings.camera.landmarksVisible ? 'bg-primary' : 'bg-muted'
                  }`}
                >
                  <div className={`w-5 h-5 rounded-full bg-white shadow transition-transform ${
                    localSettings.camera.landmarksVisible ? 'translate-x-6' : 'translate-x-0.5'
                  }`} />
                </button>
              </div>
            </div>
          )}

          {/* Model Settings */}
          {activeTab === 'model' && (
            <div className="space-y-6">
              <div>
                <label className="block text-sm font-medium mb-2">
                  Confidence Threshold: {(localSettings.model.confidenceThreshold * 100).toFixed(0)}%
                </label>
                <input
                  type="range"
                  min="50"
                  max="95"
                  value={localSettings.model.confidenceThreshold * 100}
                  onChange={(e) => setLocalSettings({
                    ...localSettings,
                    model: { ...localSettings.model, confidenceThreshold: parseInt(e.target.value) / 100 }
                  })}
                  className="w-full accent-primary"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Higher values require more confident predictions
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">
                  Sentence Buffer: {localSettings.model.sentenceBufferSize} frames
                </label>
                <input
                  type="range"
                  min="30"
                  max="150"
                  step="10"
                  value={localSettings.model.sentenceBufferSize}
                  onChange={(e) => setLocalSettings({
                    ...localSettings,
                    model: { ...localSettings.model, sentenceBufferSize: parseInt(e.target.value) }
                  })}
                  className="w-full accent-primary"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  More frames = longer sentences but higher latency
                </p>
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">Auto-Correct</p>
                  <p className="text-sm text-muted-foreground">AI grammar correction for sentences</p>
                </div>
                <button
                  onClick={() => setLocalSettings({
                    ...localSettings,
                    model: { ...localSettings.model, autoCorrect: !localSettings.model.autoCorrect }
                  })}
                  className={`w-12 h-6 rounded-full transition-colors ${
                    localSettings.model.autoCorrect ? 'bg-primary' : 'bg-muted'
                  }`}
                >
                  <div className={`w-5 h-5 rounded-full bg-white shadow transition-transform ${
                    localSettings.model.autoCorrect ? 'translate-x-6' : 'translate-x-0.5'
                  }`} />
                </button>
              </div>
            </div>
          )}

          {/* Avatar Settings */}
          {activeTab === 'avatar' && (
            <div className="space-y-6">
              <div>
                <label className="block text-sm font-medium mb-2">
                  Signing Speed: {localSettings.avatar.signSpeed}x
                </label>
                <input
                  type="range"
                  min="0.5"
                  max="2"
                  step="0.1"
                  value={localSettings.avatar.signSpeed}
                  onChange={(e) => setLocalSettings({
                    ...localSettings,
                    avatar: { ...localSettings.avatar, signSpeed: parseFloat(e.target.value) }
                  })}
                  className="w-full accent-primary"
                />
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">Show Subtitles</p>
                  <p className="text-sm text-muted-foreground">Display text while avatar signs</p>
                </div>
                <button
                  onClick={() => setLocalSettings({
                    ...localSettings,
                    avatar: { ...localSettings.avatar, showSubtitles: !localSettings.avatar.showSubtitles }
                  })}
                  className={`w-12 h-6 rounded-full transition-colors ${
                    localSettings.avatar.showSubtitles ? 'bg-primary' : 'bg-muted'
                  }`}
                >
                  <div className={`w-5 h-5 rounded-full bg-white shadow transition-transform ${
                    localSettings.avatar.showSubtitles ? 'translate-x-6' : 'translate-x-0.5'
                  }`} />
                </button>
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">Voice Output</p>
                  <p className="text-sm text-muted-foreground">Text-to-speech for avatar responses</p>
                </div>
                <button
                  onClick={() => setLocalSettings({
                    ...localSettings,
                    avatar: { ...localSettings.avatar, voiceEnabled: !localSettings.avatar.voiceEnabled }
                  })}
                  className={`w-12 h-6 rounded-full transition-colors ${
                    localSettings.avatar.voiceEnabled ? 'bg-primary' : 'bg-muted'
                  }`}
                >
                  <div className={`w-5 h-5 rounded-full bg-white shadow transition-transform ${
                    localSettings.avatar.voiceEnabled ? 'translate-x-6' : 'translate-x-0.5'
                  }`} />
                </button>
              </div>
            </div>
          )}

          {/* Display Settings */}
          {activeTab === 'display' && (
            <div className="space-y-6">
              <div>
                <label className="block text-sm font-medium mb-3">Theme</label>
                <div className="flex gap-3">
                  {[
                    { value: 'light', icon: Sun, label: 'Light' },
                    { value: 'dark', icon: Moon, label: 'Dark' },
                    { value: 'system', icon: Monitor, label: 'System' }
                  ].map((theme) => (
                    <button
                      key={theme.value}
                      onClick={() => setLocalSettings({
                        ...localSettings,
                        display: { ...localSettings.display, theme: theme.value as any }
                      })}
                      className={`flex-1 flex flex-col items-center gap-2 p-4 rounded-xl border transition-colors ${
                        localSettings.display.theme === theme.value
                          ? 'border-primary bg-primary/10'
                          : 'border-border hover:border-primary/50'
                      }`}
                    >
                      <theme.icon className="w-5 h-5" />
                      <span className="text-sm">{theme.label}</span>
                    </button>
                  ))}
                </div>
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">Notifications</p>
                  <p className="text-sm text-muted-foreground">Show recognition alerts</p>
                </div>
                <button
                  onClick={() => setLocalSettings({
                    ...localSettings,
                    display: { ...localSettings.display, notifications: !localSettings.display.notifications }
                  })}
                  className={`w-12 h-6 rounded-full transition-colors ${
                    localSettings.display.notifications ? 'bg-primary' : 'bg-muted'
                  }`}
                >
                  <div className={`w-5 h-5 rounded-full bg-white shadow transition-transform ${
                    localSettings.display.notifications ? 'translate-x-6' : 'translate-x-0.5'
                  }`} />
                </button>
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">Sound Effects</p>
                  <p className="text-sm text-muted-foreground">Audio feedback for actions</p>
                </div>
                <button
                  onClick={() => setLocalSettings({
                    ...localSettings,
                    display: { ...localSettings.display, soundEffects: !localSettings.display.soundEffects }
                  })}
                  className={`w-12 h-6 rounded-full transition-colors ${
                    localSettings.display.soundEffects ? 'bg-primary' : 'bg-muted'
                  }`}
                >
                  <div className={`w-5 h-5 rounded-full bg-white shadow transition-transform ${
                    localSettings.display.soundEffects ? 'translate-x-6' : 'translate-x-0.5'
                  }`} />
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between p-6 border-t border-border bg-muted/30">
          <button
            onClick={handleReset}
            className="flex items-center gap-2 px-4 py-2 rounded-lg hover:bg-muted transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
            Reset to Defaults
          </button>
          <div className="flex gap-3">
            <button
              onClick={onClose}
              className="px-4 py-2 rounded-lg bg-muted hover:bg-muted/80 transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              className="flex items-center gap-2 px-6 py-2 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary/90 transition-colors"
            >
              <Save className="w-4 h-4" />
              Save Changes
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SettingsModal;
