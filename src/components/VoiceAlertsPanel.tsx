import React from 'react';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Slider } from '@/components/ui/slider';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Volume2, VolumeX, Settings, Play, Square } from 'lucide-react';
import { VoiceAlertOptions } from '@/hooks/useVoiceAlerts';

interface VoiceAlertsPanelProps {
  options: VoiceAlertOptions;
  onOptionsChange: (options: Partial<VoiceAlertOptions>) => void;
  onTestVoice: () => void;
  isSpeaking: boolean;
  isSupported: boolean;
}

const VoiceAlertsPanel: React.FC<VoiceAlertsPanelProps> = ({
  options,
  onOptionsChange,
  onTestVoice,
  isSpeaking,
  isSupported,
}) => {
  if (!isSupported) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <VolumeX className="w-5 h-5 text-muted-foreground" />
            Voice Alerts
          </CardTitle>
          <CardDescription>
            Voice alerts are not supported in this browser
          </CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            Your browser doesn't support speech synthesis. Try using Chrome, Firefox, or Safari.
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Volume2 className="w-5 h-5" />
          Voice Alerts
        </CardTitle>
        <CardDescription>
          Configure audio notifications for detected objects
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Enable/Disable Switch */}
        <div className="flex items-center justify-between">
          <div className="space-y-0.5">
            <label className="text-sm font-medium">Enable Voice Alerts</label>
            <p className="text-xs text-muted-foreground">
              Turn on audio announcements for object detection
            </p>
          </div>
          <Switch
            checked={options.enabled}
            onCheckedChange={(checked) => onOptionsChange({ enabled: checked })}
          />
        </div>

        {/* Alert Type Selection */}
        <div className="space-y-2">
          <label className="text-sm font-medium">Alert Type</label>
          <Select
            value={options.alertType}
            onValueChange={(value: 'immediate' | 'summary' | 'both') => 
              onOptionsChange({ alertType: value })
            }
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="immediate">Immediate</SelectItem>
              <SelectItem value="summary">Summary</SelectItem>
              <SelectItem value="both">Both</SelectItem>
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">
            {options.alertType === 'immediate' && 'Announce each object as it\'s detected'}
            {options.alertType === 'summary' && 'Give periodic summaries of all detected objects'}
            {options.alertType === 'both' && 'Announce immediately and provide summaries'}
          </p>
        </div>

        {/* Announcement Options */}
        <div className="space-y-3">
          <label className="text-sm font-medium">Announcement Details</label>
          
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <label className="text-sm">Include Distance</label>
              <p className="text-xs text-muted-foreground">Announce object distance</p>
            </div>
            <Switch
              checked={options.announceDistance}
              onCheckedChange={(checked) => onOptionsChange({ announceDistance: checked })}
            />
          </div>

          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <label className="text-sm">Include Direction</label>
              <p className="text-xs text-muted-foreground">Announce object direction</p>
            </div>
            <Switch
              checked={options.announceDirection}
              onCheckedChange={(checked) => onOptionsChange({ announceDirection: checked })}
            />
          </div>
        </div>

        {/* Voice Settings */}
        <div className="space-y-4">
          <label className="text-sm font-medium">Voice Settings</label>
          
          {/* Volume Control */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-sm">Volume</label>
              <span className="text-xs text-muted-foreground">
                {Math.round(options.volume * 100)}%
              </span>
            </div>
            <Slider
              value={[options.volume]}
              onValueChange={(value) => onOptionsChange({ volume: value[0] })}
              max={1}
              min={0}
              step={0.1}
              className="w-full"
            />
          </div>

          {/* Speech Rate */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-sm">Speech Rate</label>
              <span className="text-xs text-muted-foreground">
                {options.rate.toFixed(1)}x
              </span>
            </div>
            <Slider
              value={[options.rate]}
              onValueChange={(value) => onOptionsChange({ rate: value[0] })}
              max={2}
              min={0.5}
              step={0.1}
              className="w-full"
            />
          </div>

          {/* Pitch */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-sm">Pitch</label>
              <span className="text-xs text-muted-foreground">
                {options.pitch.toFixed(1)}x
              </span>
            </div>
            <Slider
              value={[options.pitch]}
              onValueChange={(value) => onOptionsChange({ pitch: value[0] })}
              max={2}
              min={0.5}
              step={0.1}
              className="w-full"
            />
          </div>
        </div>

        {/* Test Button */}
        <div className="pt-2">
          <Button
            onClick={onTestVoice}
            disabled={!options.enabled || isSpeaking}
            variant="outline"
            className="w-full gap-2"
          >
            {isSpeaking ? (
              <>
                <Square className="w-4 h-4" />
                Stop Test
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                Test Voice
              </>
            )}
          </Button>
          <p className="text-xs text-muted-foreground text-center mt-2">
            Test your voice alert settings
          </p>
        </div>
      </CardContent>
    </Card>
  );
};

export default VoiceAlertsPanel;