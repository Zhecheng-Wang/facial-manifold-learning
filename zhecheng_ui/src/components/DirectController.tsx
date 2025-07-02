import React, { useRef, useMemo, useState, useEffect } from 'react';
import { useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { Sphere } from '@react-three/drei';

// Controls are toggled with Alt key instead of Shift
const CONTROL_MODIFIER_KEY = 'Alt';

interface Blendshape {
  name: string;
  vertices: Float32Array;
  center: THREE.Vector3;
  normal: THREE.Vector3;
  maxDisplacement: number;
}

interface ControllerProps {
  vertexIndex: number;
  baseVertices: Float32Array;
  normals: THREE.BufferAttribute | THREE.InterleavedBufferAttribute;
  deformation: Float32Array;
  maxLength: number;
  lengthMultiplier: number;
  isActive: boolean;
  isControlMode: boolean;
  weight: number;
  onClick: () => void;
  onPointerMove: (event: THREE.Event) => void;
  onPointerUp: () => void;
}

const Controller: React.FC<ControllerProps> = ({
  vertexIndex,
  baseVertices,
  normals,
  deformation,
  maxLength,
  lengthMultiplier,
  isActive,
  isControlMode,
  weight,
  onClick,
  onPointerMove,
  onPointerUp,
}) => {
  // Get base position from vertex
  const basePosition = new THREE.Vector3(
    baseVertices[vertexIndex * 3],
    baseVertices[vertexIndex * 3 + 1],
    baseVertices[vertexIndex * 3 + 2]
  );

  // Get normal at vertex
  const normal = new THREE.Vector3(
    normals.getX(vertexIndex),
    normals.getY(vertexIndex),
    normals.getZ(vertexIndex)
  );

  // Get deformation direction
  const deformationDir = new THREE.Vector3(
    deformation[vertexIndex * 3],
    deformation[vertexIndex * 3 + 1],
    deformation[vertexIndex * 3 + 2]
  ).normalize();

  // Combine normal and deformation for control direction
  const controlDir = new THREE.Vector3()
    .addVectors(normal.multiplyScalar(0.5), deformationDir.multiplyScalar(0.5))
    .normalize();

  // Apply length multiplier to max length
  const adjustedMaxLength = maxLength * lengthMultiplier;

  // Offset center slightly from surface
  const center = basePosition.clone().add(normal.clone().multiplyScalar(0.1));
  const endPoint = center.clone().add(controlDir.multiplyScalar(adjustedMaxLength));
  const currentPoint = center.clone().lerp(endPoint, weight);

  return (
    <group>
      {/* Base sphere - 3x larger */}
      <Sphere
        position={center.toArray()}
        args={[0.15, 16, 16]}
        onClick={onClick}
        onPointerMove={isControlMode ? onPointerMove : undefined}
        onPointerUp={onPointerUp}
      >
        <meshStandardMaterial 
          color={isActive ? "#0066cc" : "#666666"}
          transparent
          opacity={0.8}
          emissive={isActive ? "#0066cc" : "#666666"}
          emissiveIntensity={0.5}
        />
      </Sphere>

      {/* Line segment */}
      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={2}
            array={new Float32Array([
              center.x, center.y, center.z,
              endPoint.x, endPoint.y, endPoint.z,
            ])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial 
          color={isActive ? "#0066cc" : "#666666"} 
          linewidth={3}
          transparent
          opacity={0.8}
        />
      </line>

      {/* Control sphere */}
      <Sphere
        position={currentPoint.toArray()}
        args={[0.15, 16, 16]}
        onClick={onClick}
        onPointerMove={isControlMode ? onPointerMove : undefined}
        onPointerUp={onPointerUp}
      >
        <meshStandardMaterial 
          color={isActive ? "#0066cc" : "#ff0000"}
          transparent
          opacity={0.8}
          emissive={isActive ? "#0066cc" : "#ff0000"}
          emissiveIntensity={0.5}
        />
      </Sphere>
    </group>
  );
};

interface DirectControllerProps {
  blendshapes: Blendshape[];
  weights: number[];
  setWeights: (weights: number[]) => void;
  baseVertices: Float32Array;
}

const DirectController: React.FC<DirectControllerProps> = ({
  blendshapes,
  weights,
  setWeights,
  baseVertices,
}) => {
  const { camera, raycaster, gl } = useThree();
  const [activeSphere, setActiveSphere] = useState<number | null>(null);
  const spheresRef = useRef<THREE.Group>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isControlMode, setIsControlMode] = useState(false);
  const [dragPlane, setDragPlane] = useState<THREE.Plane | null>(null);
  const [lengthMultiplier, setLengthMultiplier] = useState(0.5); // Default multiplier
  
  // Reference to orbit controls to temporarily disable
  const orbitControlsRef = useRef<any>(null);
  
  // Find and store the orbit controls reference
  useEffect(() => {
    // Look for OrbitControls in the scene
    const findOrbitControls = () => {
      document.querySelectorAll('*').forEach((element: any) => {
        if (element.__r3f && element.__r3f.instance && element.__r3f.instance.enableRotate !== undefined) {
          orbitControlsRef.current = element.__r3f.instance;
        }
      });
    };
    
    // Try to find orbit controls after a short delay to ensure they're mounted
    setTimeout(findOrbitControls, 100);
  }, []);

  // Create geometry and compute normals once
  const { normals, controllerData } = useMemo(() => {
    console.log('Computing controllers for', blendshapes.length, 'blendshapes');
    
    // Create geometry to compute normals
    const tempGeometry = new THREE.BufferGeometry();
    const vertices = new Float32Array(baseVertices);
    tempGeometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    tempGeometry.computeVertexNormals();
    const normals = tempGeometry.getAttribute('normal');
    
    // Compute controller data
    const controllerData = blendshapes.map((bs, index) => {
      // Find vertices significantly affected by this blendshape
      const affectedIndices: number[] = [];
      const displacements: number[] = [];
      
      for (let i = 0; i < bs.vertices.length; i += 3) {
        const dx = bs.vertices[i];
        const dy = bs.vertices[i + 1];
        const dz = bs.vertices[i + 2];
        const displacement = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (displacement > 0.01) {
          affectedIndices.push(i / 3);
          displacements.push(displacement);
        }
      }

      if (affectedIndices.length === 0) {
        console.warn(`No significantly affected vertices found for ${bs.name}`);
        return null;
      }

      // Find vertex with maximum displacement
      const maxDispIndex = displacements.indexOf(Math.max(...displacements));
      const vertexIndex = affectedIndices[maxDispIndex];
      const maxDisplacement = displacements[maxDispIndex] * 2; // Make control line longer

      return {
        name: bs.name,
        vertexIndex,
        deformation: bs.vertices,
        maxLength: maxDisplacement
      };
    }).filter((c): c is NonNullable<typeof c> => c !== null);

    return { normals, controllerData };
  }, [blendshapes, baseVertices]);

  // Handle hotkeys for control mode and length multiplier
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === CONTROL_MODIFIER_KEY) {
        setIsControlMode(true);
        // Disable orbit controls when in control mode
        if (orbitControlsRef.current) {
          orbitControlsRef.current.enabled = false;
        }
      } else if (e.key === 'ArrowUp') {
        setLengthMultiplier(prev => Math.min(prev + 0.1, 2.0));
        console.log('Length multiplier increased to:', lengthMultiplier + 0.1);
      } else if (e.key === 'ArrowDown') {
        setLengthMultiplier(prev => Math.max(prev - 0.1, 0.1));
        console.log('Length multiplier decreased to:', lengthMultiplier - 0.1);
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.key === CONTROL_MODIFIER_KEY) {
        setIsControlMode(false);
        setIsDragging(false);
        // Re-enable orbit controls when exiting control mode
        if (orbitControlsRef.current) {
          orbitControlsRef.current.enabled = true;
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    
    // Status message for user
    console.log(`Hold ${CONTROL_MODIFIER_KEY} key to enter control mode. Use arrow up/down to adjust control length.`);
    
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [lengthMultiplier]);

  const handleSphereClick = (index: number) => {
    if (!isControlMode) return;
    
    setActiveSphere(index);
    setIsDragging(true);
    
    if (controllerData[index]) {
      // Create a drag plane perpendicular to the camera
      const controller = controllerData[index];
      const basePosition = new THREE.Vector3(
        baseVertices[controller.vertexIndex * 3],
        baseVertices[controller.vertexIndex * 3 + 1],
        baseVertices[controller.vertexIndex * 3 + 2]
      );
      
      // Get normal at vertex for proper drag plane orientation
      const normal = new THREE.Vector3(
        normals.getX(controller.vertexIndex),
        normals.getY(controller.vertexIndex),
        normals.getZ(controller.vertexIndex)
      );
      
      // Create a drag plane that contains the controller direction and is aligned with the vertex normal
      const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(
        camera.getWorldDirection(new THREE.Vector3()), 
        basePosition
      );
      setDragPlane(plane);
    }
  };

  const handlePointerMove = (event: THREE.Event) => {
    if (!isControlMode || !isDragging || activeSphere === null || !dragPlane) return;

    const { point } = event as any;
    const controller = controllerData[activeSphere];
    
    // Get base position
    const basePosition = new THREE.Vector3(
      baseVertices[controller.vertexIndex * 3],
      baseVertices[controller.vertexIndex * 3 + 1],
      baseVertices[controller.vertexIndex * 3 + 2]
    );

    // Get normal
    const normal = new THREE.Vector3(
      normals.getX(controller.vertexIndex),
      normals.getY(controller.vertexIndex),
      normals.getZ(controller.vertexIndex)
    );

    const center = basePosition.clone().add(normal.multiplyScalar(0.1));
    
    // Project the point onto our control line for more predictable control
    const controlDir = new THREE.Vector3()
      .addVectors(
        normal.clone().multiplyScalar(0.5), 
        new THREE.Vector3(
          controller.deformation[controller.vertexIndex * 3],
          controller.deformation[controller.vertexIndex * 3 + 1],
          controller.deformation[controller.vertexIndex * 3 + 2]
        ).normalize().multiplyScalar(0.5)
      )
      .normalize();
    
    // Create a line representing our control direction
    const line = new THREE.Line3(
      center,
      center.clone().add(controlDir.multiplyScalar(controller.maxLength * lengthMultiplier))
    );
    
    // Project the point onto this line
    const projectedPoint = new THREE.Vector3();
    line.closestPointToPoint(point, true, projectedPoint);
    
    // Calculate distance along the line as a percentage
    const totalLength = line.distance();
    const distance = center.distanceTo(projectedPoint);
    const weight = Math.min(1, Math.max(0, distance / totalLength));

    const newWeights = [...weights];
    newWeights[activeSphere] = weight;
    setWeights(newWeights);
  };

  const handlePointerUp = () => {
    if (isDragging) {
      setActiveSphere(null);
      setIsDragging(false);
      setDragPlane(null);
    }
  };

  return (
    <group ref={spheresRef}>
      {/* Control mode indicator */}
      {isControlMode && (
        <group position={[0, -2, 0]}>
          <mesh>
            <planeGeometry args={[1.5, 0.3]} />
            <meshBasicMaterial color="#003366" opacity={0.7} transparent />
          </mesh>
        </group>
      )}
      
      {controllerData.map((controller, index) => (
        <Controller
          key={controller.name}
          vertexIndex={controller.vertexIndex}
          baseVertices={baseVertices}
          normals={normals}
          deformation={controller.deformation}
          maxLength={controller.maxLength}
          lengthMultiplier={lengthMultiplier}
          isActive={activeSphere === index}
          isControlMode={isControlMode}
          weight={weights[index]}
          onClick={() => handleSphereClick(index)}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
        />
      ))}
    </group>
  );
};

export default DirectController; 