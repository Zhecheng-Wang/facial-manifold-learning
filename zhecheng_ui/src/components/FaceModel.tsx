import React, { useMemo, useEffect, useRef } from 'react';
import { useThree } from '@react-three/fiber';
import * as THREE from 'three';
import DirectController from './DirectController';

interface Blendshape {
  name: string;
  vertices: Float32Array;
  center: THREE.Vector3;
  normal: THREE.Vector3;
  maxDisplacement: number;
}

interface RenderSettings {
  showWireframe: boolean;
  usePhongMaterial: boolean;
  metalness: number;
  roughness: number;
  envMapIntensity: number;
}

interface FaceModelProps {
  blendshapes: Blendshape[];
  weights: number[];
  setWeights: (weights: number[]) => void;
  baseVertices: Float32Array;
  baseFaces: Uint32Array;
  renderSettings: RenderSettings;
}

const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);
const isValidNumber = (n: number) => typeof n === 'number' && !isNaN(n) && isFinite(n);

const arrayEquals = (a: number[], b: number[]) => 
  a.length === b.length && a.every((v, i) => v === b[i]);

// Add utility function for safe delta addition
const safeAddDelta = (
  dst: Float32Array,
  src: Float32Array,
  weight: number,
) => {
  const n = Math.min(dst.length, src.length);
  for (let i = 0; i < n; ++i) dst[i] += src[i] * weight;
};

const FaceModel: React.FC<FaceModelProps> = ({ 
  blendshapes, 
  weights, 
  setWeights, 
  baseVertices, 
  baseFaces,
  renderSettings 
}) => {
  const { scene } = useThree();
  const meshRef = useRef<THREE.Mesh>(null);
  const positionAttributeRef = useRef<THREE.BufferAttribute | null>(null);
  const previousWeights = useRef<number[]>([]);

  // Create geometry once with base vertices
  const geometry = useMemo(() => {
    try {
      console.log('Creating initial geometry with base vertices');
      const geometry = new THREE.BufferGeometry();
      
      // Create position buffer with base vertices
      const positions = new Float32Array(baseVertices);
      
      // Set attributes
      const positionAttribute = new THREE.BufferAttribute(positions, 3);
      geometry.setAttribute('position', positionAttribute);
      geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(baseFaces), 1));
      
      // Store reference to position attribute for updates
      positionAttributeRef.current = positionAttribute;
      
      // Compute initial normals and bounds
      geometry.computeVertexNormals();
      geometry.computeBoundingBox();
      geometry.computeBoundingSphere();

      return geometry;
    } catch (err) {
      console.error('Error creating geometry:', err);
      return new THREE.BufferGeometry();
    }
  }, [baseVertices, baseFaces]);

  // Update positions when weights change
  useEffect(() => {
    console.log('Weight update effect triggered');
    const positionAttribute = positionAttributeRef.current;
    if (!positionAttribute) {
      console.warn('No position attribute available');
      return;
    }

    // Check if weights actually changed
    const hasChanged = weights.some((w, i) => w !== previousWeights.current[i]);
    if (!hasChanged) {
      console.log('Weights unchanged, skipping update');
      return;
    }

    // Debug prints for first blendshape
    if (blendshapes.length > 0) {
      console.log('First blendshape data:');
      console.log('Name:', blendshapes[0].name);
      console.log('Vertices length:', blendshapes[0].vertices.length);
      console.log('First 9 values:', blendshapes[0].vertices.slice(0, 9));
    }

    // Diagnostic checks
    // 1. Check blend-shape delta array lengths
    blendshapes.forEach((bs) => {
      if (bs.vertices.length !== baseVertices.length) {
        console.error(
          `Blendshape ${bs.name} has ${bs.vertices.length} values, ` +
          `but the base mesh has ${baseVertices.length}.`
        );
      }
    });

    // 2. Check weights array length
    if (weights.length !== blendshapes.length) {
      console.error(
        `weights has length ${weights.length} but there are ` +
        `${blendshapes.length} blend-shapes.`
      );
    }

    // 3. Check for non-finite weights
    weights.forEach((w, i) => {
      if (!isFinite(w)) {
        console.error(`weights[${i}] =`, w, 'is not finite');
      }
    });

    console.log('Updating geometry with weights:', weights);
    
    // Start with base vertices
    const newPositions = new Float32Array(baseVertices);
    console.log('Initial positions:', newPositions.slice(0, 6)); // Log first 2 vertices
    
    // Apply blendshape deltas using matrix multiplication
    const numVertices = baseVertices.length / 3;
    const numBlendshapes = blendshapes.length;
    
    // Create a matrix of all blendshape deltas (numBlendshapes × numVertices*3)
    const deltaMatrix = new Float32Array(numBlendshapes * numVertices * 3);
    blendshapes.forEach((bs, i) => {
      // Verify the vertex data is in the correct order [x0,y0,z0, x1,y1,z1, ...]
      const vertices = bs.vertices;
      for (let v = 0; v < numVertices; v++) {
        for (let c = 0; c < 3; c++) {
          deltaMatrix[i * numVertices * 3 + v * 3 + c] = vertices[v * 3 + c];
        }
      }
    });
    
    // Apply weights using matrix multiplication
    for (let v = 0; v < numVertices; v++) {
      for (let c = 0; c < 3; c++) {
        let sum = 0;
        for (let b = 0; b < numBlendshapes; b++) {
          sum += weights[b] * deltaMatrix[b * numVertices * 3 + v * 3 + c];
        }
        newPositions[v * 3 + c] += sum;
      }
    }

    // Check for NaN values after application
    for (let i = 0; i < newPositions.length; ++i) {
      if (!isFinite(newPositions[i])) {
        console.error(
          'Non-finite value at vertex', Math.floor(i / 3),
          ['x', 'y', 'z'][i % 3]
        );
        throw new Error('Bad position value');
      }
    }

    console.log('Final positions after blendshapes:', newPositions.slice(0, 6)); // Log first 2 vertices
    
    // Update position attribute with new positions
    positionAttribute.array.set(newPositions);
    positionAttribute.needsUpdate = true;
    
    // Update geometry
    geometry.computeVertexNormals();
    geometry.computeBoundingSphere(); // Keep culling sane

    // Store current weights
    previousWeights.current = [...weights];
  }, [weights, blendshapes, baseVertices, geometry]);

  // Create materials
  const materials = useMemo(() => {
    const mainMaterial = renderSettings.usePhongMaterial
      ? new THREE.MeshPhongMaterial({
          color: 0xffffff,
          shininess: 100,
          specular: 0x444444,
        })
      : new THREE.MeshStandardMaterial({
          color: 0xffffff,
          metalness: renderSettings.metalness,
          roughness: renderSettings.roughness,
          envMapIntensity: renderSettings.envMapIntensity,
        });

    const wireframeMaterial = new THREE.MeshBasicMaterial({
      color: 0x000000,
      wireframe: true,
      transparent: true,
      opacity: 0.1,
    });

    return [mainMaterial, wireframeMaterial];
  }, [renderSettings]);

  return (
    <group>
      <mesh ref={meshRef} geometry={geometry} material={materials[0]}>
        {renderSettings.showWireframe && (
          <mesh geometry={geometry} material={materials[1]} />
        )}
      </mesh>
      <DirectController
        blendshapes={blendshapes}
        weights={weights}
        setWeights={setWeights}
        baseVertices={baseVertices}
      />
    </group>
  );
};

export default FaceModel; 