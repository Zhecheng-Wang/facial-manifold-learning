import React, { useState } from 'react';
import { MantineProvider } from '@mantine/core';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stats, Environment } from '@react-three/drei';
import { useControls, folder, LevaPanel, useCreateStore, Leva } from 'leva';
import FaceModel from './components/FaceModel';
import { useBlendshapes } from './hooks/useBlendshapes';
import { BlendshapeSliders } from './components/BlendshapeSliders';

interface BlendshapeControls {
  [key: string]: number;
}

const App: React.FC = () => {
  const { blendshapes, weights, setWeights, loading, error, baseVertices, baseFaces } = useBlendshapes();
  
  // Create separate stores for each panel
  const renderStore = useCreateStore();
  const blendshapeStore = useCreateStore();

  // Visualization controls with Leva (left panel)
  const { showWireframe, usePhongMaterial, metalness, roughness, envMapIntensity } = useControls('Rendering', {
    Material: folder({
      usePhongMaterial: { value: true, label: 'Use Phong Material' },
      showWireframe: { value: true, label: 'Show Wireframe' },
      metalness: { value: 0.5, min: 0, max: 1, label: 'Metalness' },
      roughness: { value: 0.2, min: 0, max: 1, label: 'Roughness' },
      envMapIntensity: { value: 1, min: 0, max: 5, label: 'Environment Map' },
    })
  }, { store: renderStore });

  // Blendshape controls with Leva (right panel)
  const blendshapeControls = useControls(
    'Blendshapes',
    blendshapes?.reduce<Record<string, { value: number; min: number; max: number; label: string }>>((acc, bs) => ({
      ...acc,
      [bs.name]: { value: 0, min: 0, max: 1, label: bs.name }
    }), {}) || {},
    { store: blendshapeStore }
  ) as BlendshapeControls;

  // Update weights when blendshape controls change
  React.useEffect(() => {
    if (blendshapes && blendshapeControls) {
      const newWeights = blendshapes.map((bs) => blendshapeControls[bs.name] || 0);
      console.log('Updating weights:', newWeights); // Debug log
      setWeights(newWeights);
    }
  }, [blendshapeControls, blendshapes, setWeights]);

  if (loading) {
    return (
      <MantineProvider>
        <div className="loading-container">
          <div className="uk-text-center">
            <div className="uk-spinner" uk-spinner="ratio: 3"></div>
            <p>Loading face data...</p>
          </div>
        </div>
      </MantineProvider>
    );
  }

  if (error) {
    return <div>Error: {error}</div>;
  }

  return (
    <MantineProvider>
      <Leva hidden /> {/* Root Leva component */}
      
      {/* Rendering controls panel */}
      <div style={{ position: 'fixed', left: 16, bottom: 16, zIndex: 1000, width: '300px' }}>
        <LevaPanel 
          store={renderStore}
          fill
          flat
          titleBar={false}
        />
      </div>
      
      {/* Blendshapes panel */}
      <div style={{ position: 'fixed', right: 16, top: 16, zIndex: 1000, width: '400px' }}>
        <LevaPanel 
          store={blendshapeStore}
          fill
          flat
          titleBar={false}
        />
      </div>

      {/* Mount blendshape controls only after data is loaded */}
      {blendshapes && (
        <BlendshapeSliders
          blendshapes={blendshapes}
          setWeights={setWeights}
          store={blendshapeStore}
        />
      )}

      <div style={{ width: '100vw', height: '100vh', background: '#1a1a1a' }}>
        <Canvas camera={{ position: [30, 5, 50], fov: 45 }}>
          <color attach="background" args={['#1a1a1a']} />
          <Stats />
          
          <OrbitControls 
            makeDefault
            enableDamping
            dampingFactor={0.1}
            minDistance={30}
            maxDistance={150}
            target={[0, 0, 0]}
            minPolarAngle={Math.PI / 6}
            maxPolarAngle={Math.PI * 5/6}
          />

          <Environment preset="apartment" />
          <ambientLight intensity={0.5} />
          <directionalLight position={[2, 2, 5]} intensity={1} castShadow />
          
          {!loading && !error && baseVertices && baseFaces && (
            <FaceModel
              blendshapes={blendshapes}
              weights={weights}
              setWeights={setWeights}
              baseVertices={baseVertices}
              baseFaces={baseFaces}
              renderSettings={{
                showWireframe,
                usePhongMaterial,
                metalness,
                roughness,
                envMapIntensity
              }}
            />
          )}
        </Canvas>
      </div>
    </MantineProvider>
  );
};

export default App; 