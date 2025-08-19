import { Image as ImageIcon, Mic, Type, Settings } from "lucide-react";
import { Button } from "@/components/ui/button";

interface RadialMenuProps {
  onUploadImage: () => void;
  onUploadVoice: () => void;
  onEnterText: () => void;
  onSettings: () => void;
}

export const RadialMenu = ({ 
  onUploadImage, 
  onUploadVoice, 
  onEnterText, 
  onSettings 
}: RadialMenuProps) => {
  return (
    <div className="fixed bottom-6 right-6 z-50">
      <div className="relative">
        {/* Main button */}
        <Button 
          size="lg" 
          className="w-16 h-16 rounded-full glass-button shadow-lg"
          onClick={onSettings}
        >
          <Settings size={24} />
        </Button>
        
        {/* Radial menu items */}
        <div className="absolute bottom-0 right-0 space-y-2">
          <Button 
            size="sm" 
            className="radial-menu-item"
            onClick={onUploadImage}
          >
            <ImageIcon size={20} />
          </Button>
          
          <Button 
            size="sm" 
            className="radial-menu-item"
            onClick={onUploadVoice}
          >
            <Mic size={20} />
          </Button>
          
          <Button 
            size="sm" 
            className="radial-menu-item"
            onClick={onEnterText}
          >
            <Type size={20} />
          </Button>
        </div>
      </div>
    </div>
  );
};

