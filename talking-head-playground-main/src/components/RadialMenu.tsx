import { useState } from "react";
import { Plus, Image as ImageIcon, Mic, Type, Settings } from "lucide-react";
import { cn } from "@/lib/utils";

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
  const [isOpen, setIsOpen] = useState(false);

  const menuItems = [
    { icon: ImageIcon, label: "Upload Image", action: onUploadImage, color: "from-blue-500 to-purple-500" },
    { icon: Mic, label: "Upload Voice", action: onUploadVoice, color: "from-green-500 to-teal-500" },
    { icon: Type, label: "Enter Text", action: onEnterText, color: "from-pink-500 to-red-500" },
    { icon: Settings, label: "Settings", action: onSettings, color: "from-gray-500 to-slate-500" },
  ];

  return (
    <div className="fixed bottom-8 right-8 z-50">
      {/* Menu Items */}
      {isOpen && (
        <div className="absolute bottom-16 right-0 space-y-3">
          {menuItems.map((item, index) => (
            <div
              key={item.label}
              className={cn(
                "radial-menu-item opacity-0 translate-y-4",
                isOpen && "opacity-100 translate-y-0"
              )}
              style={{
                transitionDelay: `${index * 100}ms`,
                background: `linear-gradient(135deg, ${item.color.split(' ')[0].split('-')[1]}, ${item.color.split(' ')[1].split('-')[1]})`
              }}
              onClick={() => {
                item.action();
                setIsOpen(false);
              }}
            >
              <item.icon size={24} className="text-white" />
            </div>
          ))}
        </div>
      )}

      {/* Main Button */}
      <button
        className={cn(
          "w-16 h-16 glass-button rounded-full flex items-center justify-center transition-all duration-300",
          isOpen && "rotate-45 scale-110"
        )}
        onClick={() => setIsOpen(!isOpen)}
      >
        <Plus size={28} className="text-white" />
      </button>
    </div>
  );
};