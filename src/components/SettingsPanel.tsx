import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Settings, ChevronDown, ChevronRight } from 'lucide-react';
import VoiceAlertsPanel from './VoiceAlertsPanel';
import { VoiceAlertOptions } from '@/hooks/useVoiceAlerts';

interface SettingsPanelProps {
  voiceAlertOptions: VoiceAlertOptions;
  onVoiceAlertOptionsChange: (options: Partial<VoiceAlertOptions>) => void;
  onTestVoice: () => void;
  isSpeaking: boolean;
  isVoiceSupported: boolean;
}

const SettingsPanel: React.FC<SettingsPanelProps> = ({
  voiceAlertOptions,
  onVoiceAlertOptionsChange,
  onTestVoice,
  isSpeaking,
  isVoiceSupported,
}) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <Card className="w-full">
      <Collapsible open={isOpen} onOpenChange={setIsOpen}>
        <CollapsibleTrigger asChild>
          <Button
            variant="ghost"
            className="w-full justify-between p-4 h-auto"
          >
            <div className="flex items-center gap-2">
              <Settings className="w-5 h-5" />
              <span className="font-semibold">Settings</span>
            </div>
            {isOpen ? (
              <ChevronDown className="w-4 h-4" />
            ) : (
              <ChevronRight className="w-4 h-4" />
            )}
          </Button>
        </CollapsibleTrigger>
        
        <CollapsibleContent>
          <CardContent className="pt-0 space-y-4">
            {/* Voice Alerts Section */}
            <VoiceAlertsPanel
              options={voiceAlertOptions}
              onOptionsChange={onVoiceAlertOptionsChange}
              onTestVoice={onTestVoice}
              isSpeaking={isSpeaking}
              isSupported={isVoiceSupported}
            />

            {/* Future Settings Sections */}
            <div className="space-y-4">
              <div className="text-center p-4 border-2 border-dashed border-muted-foreground/25 rounded-lg">
                <Settings className="w-8 h-8 mx-auto mb-2 text-muted-foreground/50" />
                <p className="text-sm text-muted-foreground">
                  More settings coming soon...
                </p>
                <p className="text-xs text-muted-foreground/75 mt-1">
                  Detection history, custom models, export options
                </p>
              </div>
            </div>
          </CardContent>
        </CollapsibleContent>
      </Collapsible>
    </Card>
  );
};

export default SettingsPanel;