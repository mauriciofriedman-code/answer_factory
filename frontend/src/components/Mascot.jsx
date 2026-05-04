export default function Mascot({ mood = 'normal' }) {
  return (
    <div className="mascot">
      <div className={`robot ${mood}`}>
        <div className="robot-antenna" />
        <div className="robot-head">
          <div className="robot-eyes">
            <div className="robot-eye" />
            <div className="robot-eye" />
          </div>
          <div className="robot-mouth" />
        </div>
        <div className="robot-body" />
      </div>
    </div>
  );
}
