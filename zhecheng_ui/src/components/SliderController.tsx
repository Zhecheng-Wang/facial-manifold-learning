import React from 'react';
import { Slider, Text, ScrollArea, Title } from '@mantine/core';

interface Blendshape {
  name: string;
  vertices: Float32Array;
  center: THREE.Vector3;
  normal: THREE.Vector3;
  maxDisplacement: number;
}

interface SliderControllerProps {
  blendshapes: Blendshape[];
  weights: number[];
  setWeights: (weights: number[]) => void;
}

const SliderController: React.FC<SliderControllerProps> = ({
  blendshapes,
  weights,
  setWeights,
}) => {
  const handleSliderChange = (index: number, value: number) => {
    const newWeights = [...weights];
    newWeights[index] = value;
    setWeights(newWeights);
  };

  // Group blendshapes by their first letter for better organization
  const groupedBlendshapes = React.useMemo(() => {
    const groups: { [key: string]: Blendshape[] } = {};
    blendshapes.forEach((bs, globalIndex) => {
      const group = bs.name[0].toUpperCase();
      if (!groups[group]) {
        groups[group] = [];
      }
      groups[group].push({ ...bs, globalIndex });
    });
    return groups;
  }, [blendshapes]);

  return (
    <div className="control-panel">
      <div className="control-panel-header">
        <Title order={3}>Blendshape Controls</Title>
      </div>
      <ScrollArea className="control-panel-content">
        {Object.entries(groupedBlendshapes).map(([group, groupBlendshapes]) => (
          <div key={group} className="slider-group">
            <div className="slider-group-header">
              <Title order={4}>{group}</Title>
            </div>
            {groupBlendshapes.map((bs: any) => (
              <div key={bs.name} className="slider-row">
                <div className="slider-label-row">
                  <Text className="slider-label" lineClamp={1} title={bs.name}>
                    {bs.name}
                  </Text>
                  <Text className="slider-value">
                    {weights[bs.globalIndex].toFixed(2)}
                  </Text>
                </div>
                <Slider
                  min={0}
                  max={1}
                  step={0.01}
                  value={weights[bs.globalIndex]}
                  onChange={(value) => handleSliderChange(bs.globalIndex, value)}
                  size="sm"
                  color="blue"
                />
              </div>
            ))}
          </div>
        ))}
      </ScrollArea>
    </div>
  );
};

export default SliderController; 