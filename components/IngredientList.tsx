"use client";

interface Props {
  ingredients: string[];
}

export default function IngredientList({ ingredients }: Props) {
  if (ingredients.length === 0) return null;

  return (
    <div className="space-y-2">
      <h3 className="text-xs font-semibold text-stone-400 uppercase tracking-wider">
        Key Ingredients to Look For
      </h3>
      <div className="flex flex-wrap gap-2">
        {ingredients.map((ing) => (
          <span
            key={ing}
            className="px-3 py-1 rounded-full bg-stone-100 text-stone-700 border border-stone-200 text-sm font-medium capitalize"
          >
            {ing}
          </span>
        ))}
      </div>
    </div>
  );
}
